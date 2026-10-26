#include "local_indexed_kv.h"

#include <algorithm>
#include <array>
#include <atomic>
#include <chrono>
#include <cmath>
#include <cstring>
#include <fstream>
#include <limits>
#include <stdexcept>
#include <unordered_map>
#include <unordered_set>

#include <unistd.h>

#if (defined(__x86_64__) || defined(__i386__)) && \
    (defined(__GNUC__) || defined(__clang__))
#include <immintrin.h>
#define RECSTORE_ADAM_TARGET_AVX2 1
#else
#define RECSTORE_ADAM_TARGET_AVX2 0
#endif

// Scalar and SIMD updates must retain the same intermediate rounding points.
#if defined(__FAST_MATH__)
#error "RecStore exact AdamW updates require fast-math to be disabled"
#endif
#if defined(__clang__)
#pragma clang fp contract(off)
#elif defined(__GNUC__)
#pragma GCC optimize("fp-contract=off")
#endif

#include "storage/index/utils/hash.h"

namespace recstore::python {

namespace {

inline uint64_t QuantaPartitionOrder(uint64_t key) {
  return (key << 48) | (key >> 16);
}

bool IsStrictlyPartitionOrdered(const Int64KeyView& keys) {
  if (keys.size() < 2)
    return true;
  uint64_t previous = QuantaPartitionOrder(keys[0]);
  for (size_t i = 1; i < keys.size(); ++i) {
    const uint64_t current = QuantaPartitionOrder(keys[i]);
    if (current <= previous)
      return false;
    previous = current;
  }
  return true;
}

void AppendMergedGradient(std::vector<uint64_t>* keys,
                          std::vector<float>* grads,
                          uint64_t key,
                          const float* grad,
                          size_t dim) {
  keys->push_back(key);
  const size_t offset = grads->size();
  grads->resize(offset + dim);
  std::memcpy(grads->data() + offset, grad, dim * sizeof(float));
}

bool MergeTwoGradientBatches(const Int64KeyView& keys_a,
                             const float* grads_a,
                             const Int64KeyView& keys_b,
                             const float* grads_b,
                             size_t dim,
                             std::vector<uint64_t>* merged_keys,
                             std::vector<float>* merged_grads) {
  merged_keys->clear();
  merged_grads->clear();
  merged_keys->reserve(keys_a.size() + keys_b.size());
  merged_grads->reserve((keys_a.size() + keys_b.size()) * dim);

  const bool ordered = IsStrictlyPartitionOrdered(keys_a) &&
                       IsStrictlyPartitionOrdered(keys_b);
  if (ordered) {
    size_t a = 0;
    size_t b = 0;
    while (a < keys_a.size() || b < keys_b.size()) {
      if (b == keys_b.size() ||
          (a < keys_a.size() &&
           QuantaPartitionOrder(keys_a[a]) <
               QuantaPartitionOrder(keys_b[b]))) {
        AppendMergedGradient(
            merged_keys, merged_grads, keys_a[a], grads_a + a * dim, dim);
        ++a;
      } else if (a == keys_a.size() ||
                 QuantaPartitionOrder(keys_b[b]) <
                     QuantaPartitionOrder(keys_a[a])) {
        AppendMergedGradient(
            merged_keys, merged_grads, keys_b[b], grads_b + b * dim, dim);
        ++b;
      } else {
        // The partition transform is bijective, so equal order keys are equal
        // semantic IDs. Preserve the old concatenation accumulation order.
        AppendMergedGradient(
            merged_keys, merged_grads, keys_a[a], grads_a + a * dim, dim);
        float* destination =
            merged_grads->data() + (merged_keys->size() - 1) * dim;
        const float* source = grads_b + b * dim;
        for (size_t column = 0; column < dim; ++column)
          destination[column] += source[column];
        ++a;
        ++b;
      }
    }
    return true;
  }

  // Direct callers are not required to use Quanta's partition ordering. Keep
  // the binding exact for those inputs while reserving the linear merge for
  // the production path.
  std::unordered_map<uint64_t, size_t> positions;
  positions.reserve(keys_a.size() + keys_b.size());
  auto accumulate = [&](const Int64KeyView& keys, const float* grads) {
    for (size_t row = 0; row < keys.size(); ++row) {
      const auto [iterator, inserted] =
          positions.emplace(keys[row], merged_keys->size());
      if (inserted) {
        AppendMergedGradient(
            merged_keys, merged_grads, keys[row], grads + row * dim, dim);
        continue;
      }
      float* destination = merged_grads->data() + iterator->second * dim;
      const float* source = grads + row * dim;
      for (size_t column = 0; column < dim; ++column)
        destination[column] += source[column];
    }
  };
  accumulate(keys_a, grads_a);
  accumulate(keys_b, grads_b);
  return false;
}

bool AdamAvx2Available() {
#if RECSTORE_ADAM_TARGET_AVX2
  static const bool available = __builtin_cpu_supports("avx2");
  return available;
#else
  return false;
#endif
}

inline void UpdateAdamRowScalar(float* embedding,
                                float* exp_avg,
                                float* exp_avg_sq,
                                const float* grad,
                                int64_t dim,
                                double beta1,
                                double beta2,
                                double eps,
                                double weight_scale,
                                float alpha) {
  for (int64_t c = 0; c < dim; ++c) {
    const size_t col       = static_cast<size_t>(c);
    const float grad_value = grad[col];
    embedding[col] = static_cast<float>(
        static_cast<double>(embedding[col]) * weight_scale);
    exp_avg[col] = static_cast<float>(
        beta1 * static_cast<double>(exp_avg[col]) +
        (1.0 - beta1) * static_cast<double>(grad_value));
    exp_avg_sq[col] = static_cast<float>(
        beta2 * static_cast<double>(exp_avg_sq[col]) +
        (1.0 - beta2) * static_cast<double>(grad_value) *
            static_cast<double>(grad_value));
    embedding[col] = static_cast<float>(
        static_cast<double>(embedding[col]) +
        static_cast<double>(alpha) *
            (static_cast<double>(exp_avg[col]) /
             (static_cast<double>(std::sqrt(exp_avg_sq[col])) + eps)));
  }
}

#if RECSTORE_ADAM_TARGET_AVX2
__attribute__((target("avx2,no-fma"))) void
UpdateAdamRowAvx2(float* embedding,
                  float* exp_avg,
                  float* exp_avg_sq,
                  const float* grad,
                  int64_t dim,
                  double beta1,
                  double beta2,
                  double eps,
                  double weight_scale,
                  float alpha) {
  const __m256d b1 = _mm256_set1_pd(beta1);
  const __m256d b2 = _mm256_set1_pd(beta2);
  const __m256d one_minus_b1 = _mm256_set1_pd(1.0 - beta1);
  const __m256d one_minus_b2 = _mm256_set1_pd(1.0 - beta2);
  const __m256d scale = _mm256_set1_pd(weight_scale);
  const __m256d alpha_d = _mm256_set1_pd(static_cast<double>(alpha));
  const __m256d epsilon = _mm256_set1_pd(eps);
  int64_t c = 0;
  for (; c + 4 <= dim; c += 4) {
    const __m256d g = _mm256_cvtps_pd(_mm_loadu_ps(grad + c));
    const __m128 decayed = _mm256_cvtpd_ps(_mm256_mul_pd(
        _mm256_cvtps_pd(_mm_loadu_ps(embedding + c)), scale));
    const __m128 m = _mm256_cvtpd_ps(_mm256_add_pd(
        _mm256_mul_pd(b1, _mm256_cvtps_pd(_mm_loadu_ps(exp_avg + c))),
        _mm256_mul_pd(one_minus_b1, g)));
    // Preserve ((1 - beta2) * g) * g, including the intermediate rounding.
    const __m128 v = _mm256_cvtpd_ps(_mm256_add_pd(
        _mm256_mul_pd(b2, _mm256_cvtps_pd(_mm_loadu_ps(exp_avg_sq + c))),
        _mm256_mul_pd(_mm256_mul_pd(one_minus_b2, g), g)));
    // sqrt must round to float before the denominator is promoted to double.
    const __m256d denominator = _mm256_add_pd(
        _mm256_cvtps_pd(_mm_sqrt_ps(v)), epsilon);
    const __m128 updated = _mm256_cvtpd_ps(_mm256_add_pd(
        _mm256_cvtps_pd(decayed),
        _mm256_mul_pd(alpha_d,
                      _mm256_div_pd(_mm256_cvtps_pd(m), denominator))));
    _mm_storeu_ps(exp_avg + c, m);
    _mm_storeu_ps(exp_avg_sq + c, v);
    _mm_storeu_ps(embedding + c, updated);
  }
  UpdateAdamRowScalar(embedding + c, exp_avg + c, exp_avg_sq + c, grad + c,
                      dim - c, beta1, beta2, eps, weight_scale, alpha);
}
#endif

} // namespace

py::dict LocalIndexedKV::SetAdamExecution(bool vectorized,
                                         int prefetch_distance) {
  if (prefetch_distance < 0 || prefetch_distance > 64) {
    throw std::invalid_argument("Adam prefetch_distance must be in [0, 64]");
  }
  {
    // An update releases the GIL while holding mu; do not wait for mu with
    // the GIL held or the update cannot finish and release its lock.
    py::gil_scoped_release nogil;
    std::lock_guard<std::mutex> lock(mu);
    adam_vectorized = vectorized;
    adam_prefetch_distance = prefetch_distance;
  }
  py::dict config;
  config["vectorized_requested"] = vectorized;
  config["avx2_available"] = AdamAvx2Available();
  config["vectorized_active"] = vectorized && dim >= 4 && AdamAvx2Available();
  config["prefetch_distance"] = prefetch_distance;
  return config;
}

void LocalIndexedKV::ClearAdamSlotRow(int64_t row_index) {
  const size_t block = static_cast<size_t>(row_index / kAdamSlotBlockRows);
  if (block >= adam_slot_blocks.size()) {
    return;
  }
  std::memset(AdamSlotPtr(row_index, false),
              0,
              static_cast<size_t>(dim) * sizeof(float));
  std::memset(AdamSlotPtr(row_index, true),
              0,
              static_cast<size_t>(dim) * sizeof(float));
}

void LocalIndexedKV::EnsureAdamSlotRows(int64_t rows) {
  if (rows < 0 || rows > capacity) {
    throw std::runtime_error("Adam slot row count exceeds engine capacity");
  }
  const size_t required_blocks =
      static_cast<size_t>((rows + kAdamSlotBlockRows - 1) / kAdamSlotBlockRows);
  const size_t existing_blocks = adam_slot_blocks.size();
  if (existing_blocks >= required_blocks) {
    return;
  }

  adam_slot_blocks.reserve(required_blocks);
  std::vector<std::unique_ptr<AdamSlotBlock>> new_blocks;
  new_blocks.reserve(required_blocks - existing_blocks);
  while (existing_blocks + new_blocks.size() < required_blocks) {
    new_blocks.push_back(std::make_unique<AdamSlotBlock>(dim, false));
  }

  const size_t total_rows =
      new_blocks.size() * static_cast<size_t>(kAdamSlotBlockRows);
  const int workers = EffectiveWorkers(total_rows);
#pragma omp parallel for num_threads(workers) schedule(static)
  for (int worker = 0; worker < workers; ++worker) {
    const size_t begin = total_rows * static_cast<size_t>(worker) /
                         static_cast<size_t>(workers);
    const size_t end = total_rows * static_cast<size_t>(worker + 1) /
                       static_cast<size_t>(workers);
    for (size_t block = begin / static_cast<size_t>(kAdamSlotBlockRows);
         block <= (end - 1) / static_cast<size_t>(kAdamSlotBlockRows);
         ++block) {
      const size_t block_begin =
          block * static_cast<size_t>(kAdamSlotBlockRows);
      const size_t local_begin = begin > block_begin ? begin - block_begin : 0;
      const size_t local_end = std::min(
          end - block_begin, static_cast<size_t>(kAdamSlotBlockRows));
      const size_t offset = local_begin * static_cast<size_t>(dim);
      const size_t bytes =
          (local_end - local_begin) * static_cast<size_t>(dim) * sizeof(float);
      std::memset(new_blocks[block]->exp_avg.get() + offset, 0, bytes);
      std::memset(new_blocks[block]->exp_avg_sq.get() + offset, 0, bytes);
    }
  }
  for (auto& block : new_blocks) {
    adam_slot_blocks.push_back(std::move(block));
  }
}

float* LocalIndexedKV::AdamSlotPtr(int64_t row_index, bool squared) {
  const size_t block = static_cast<size_t>(row_index / kAdamSlotBlockRows);
  const size_t row   = static_cast<size_t>(row_index % kAdamSlotBlockRows);
  float* values      = squared ? adam_slot_blocks[block]->exp_avg_sq.get()
                               : adam_slot_blocks[block]->exp_avg.get();
  return values + row * static_cast<size_t>(dim);
}

void LocalIndexedKV::EnableAdamSlots() {
  py::gil_scoped_release nogil;
  std::lock_guard<std::mutex> lock(mu);
  EnsureAdamSlotRows(next_row_index);
  eager_adam_slots = true;
}

bool LocalIndexedKV::ApplySgdUpdateParallel(
    const Int64KeyView& keys, const float* grads, float lr) {
  const int workers = EffectiveWorkers(keys.size());
  std::vector<uint8_t> worker_ok(static_cast<size_t>(workers), 1);
#pragma omp parallel for num_threads(workers) schedule(static)
  for (int worker = 0; worker < workers; ++worker) {
    const size_t begin = keys.size() * static_cast<size_t>(worker) /
                         static_cast<size_t>(workers);
    const size_t end = keys.size() * static_cast<size_t>(worker + 1) /
                       static_cast<size_t>(workers);
    if (begin == end)
      continue;
    worker_ok[static_cast<size_t>(worker)] =
        kv->ApplySgdUpdateFlatRange(
            base::ConstArray<uint64_t>(
                keys.data() + begin, static_cast<int>(end - begin)),
            grads + begin * static_cast<size_t>(dim),
            static_cast<int64_t>(end - begin),
            storage_dim,
            static_cast<int64_t>(sizeof(int64_t) / sizeof(float)),
            dim,
            lr,
            static_cast<unsigned>(worker))
            ? 1
            : 0;
  }
  return std::all_of(worker_ok.begin(), worker_ok.end(), [](uint8_t ok) {
    return ok != 0;
  });
}

bool LocalIndexedKV::ApplyAdamWTFUpdateParallel(
    const Int64KeyView& keys,
    const float* grads,
    double lr,
    double beta1,
    double beta2,
    double eps,
    double weight_decay,
    AdamUpdateStats* stats) {
  using clock            = std::chrono::steady_clock;
  const int workers      = EffectiveWorkers(keys.size());
  const size_t row_bytes = static_cast<size_t>(storage_dim) * sizeof(float);
  std::vector<BaseKV::BatchGetFlatStats> worker_stats(
      static_cast<size_t>(workers));
  std::vector<uint8_t> worker_ok(static_cast<size_t>(workers), 1);

  const auto get_start = clock::now();
#pragma omp parallel for num_threads(workers) schedule(static)
  for (int worker = 0; worker < workers; ++worker) {
    const size_t begin = keys.size() * static_cast<size_t>(worker) /
                         static_cast<size_t>(workers);
    const size_t end = keys.size() * static_cast<size_t>(worker + 1) /
                       static_cast<size_t>(workers);
    if (begin == end)
      continue;
    auto& local_rows = worker_rows_scratch[static_cast<size_t>(worker)];
    const bool ok    = kv->BatchGetDirectFixedRows(
        base::ConstArray<uint64_t>(
            keys.data() + begin, static_cast<int>(end - begin)),
        static_cast<int64_t>(end - begin),
        storage_dim,
        static_cast<unsigned>(worker),
        &local_rows,
        &worker_stats[static_cast<size_t>(worker)]);
    if (!ok || local_rows.size() != end - begin) {
      worker_ok[static_cast<size_t>(worker)] = 0;
      continue;
    }
  }
  const auto get_end = clock::now();
  if (!std::all_of(worker_ok.begin(), worker_ok.end(), [](uint8_t ok) {
        return ok != 0;
      })) {
    return false;
  }

  auto& row_indices = row_indices_scratch;
  row_indices.resize(keys.size());
  int invalid_row       = 0;
  int64_t max_row_index = -1;
#pragma omp parallel for num_threads(workers) schedule(static)                 \
    reduction(| : invalid_row) reduction(max : max_row_index)
  for (int worker = 0; worker < workers; ++worker) {
    const size_t begin = keys.size() * static_cast<size_t>(worker) /
                         static_cast<size_t>(workers);
    const auto& local_rows = worker_rows_scratch[static_cast<size_t>(worker)];
    for (size_t local = 0; local < local_rows.size(); ++local) {
      const auto& row = local_rows[local];
      if (row.missing || row.data == nullptr || row.size != row_bytes) {
        invalid_row = 1;
        continue;
      }
      int64_t row_index = -1;
      std::memcpy(&row_index, row.data, sizeof(int64_t));
      if (row_index < 0 || row_index >= capacity) {
        invalid_row = 1;
        continue;
      }
      row_indices[begin + local] = row_index;
      max_row_index              = std::max(max_row_index, row_index);
    }
  }
  if (invalid_row != 0)
    return false;

  const auto alloc_start = clock::now();
  EnsureAdamSlotRows(max_row_index + 1);
  const auto alloc_end = clock::now();

  // Match Quanta's float beta tensors multiplied by double hyperparameters,
  // then its double learning-rate expression with one final float rounding.
  const float next_beta1_t = static_cast<float>(
      static_cast<double>(adam_beta1_t) * beta1);
  const float next_beta2_t = static_cast<float>(
      static_cast<double>(adam_beta2_t) * beta2);
  const float alpha = static_cast<float>(
      -lr / static_cast<double>(1.0f - next_beta1_t) *
      static_cast<double>(std::sqrt(1.0f - next_beta2_t)));
  const double weight_scale = 1.0 - weight_decay;
  const bool use_avx2 = adam_vectorized && dim >= 4 && AdamAvx2Available();
  const auto math_start     = clock::now();
#pragma omp parallel for num_threads(workers) schedule(static)
  for (int worker = 0; worker < workers; ++worker) {
    const size_t begin = keys.size() * static_cast<size_t>(worker) /
                         static_cast<size_t>(workers);
    const auto& local_rows = worker_rows_scratch[static_cast<size_t>(worker)];
    for (size_t local = 0; local < local_rows.size(); ++local) {
      if (adam_prefetch_distance > 0 &&
          static_cast<size_t>(adam_prefetch_distance) < local_rows.size() - local) {
        const size_t ahead = local + static_cast<size_t>(adam_prefetch_distance);
#if defined(__GNUC__) || defined(__clang__)
        __builtin_prefetch(local_rows[ahead].data + sizeof(int64_t), 1, 1);
        __builtin_prefetch(AdamSlotPtr(row_indices[begin + ahead], false), 1, 1);
        __builtin_prefetch(AdamSlotPtr(row_indices[begin + ahead], true), 1, 1);
#endif
      }
      const size_t row_pos = begin + local;
      float* embedding     = reinterpret_cast<float*>(
          const_cast<char*>(local_rows[local].data) + sizeof(int64_t));
      float* exp_avg    = AdamSlotPtr(row_indices[row_pos], false);
      float* exp_avg_sq = AdamSlotPtr(row_indices[row_pos], true);
      const float* grad = grads + row_pos * static_cast<size_t>(dim);
#if RECSTORE_ADAM_TARGET_AVX2
      if (use_avx2) {
        UpdateAdamRowAvx2(embedding, exp_avg, exp_avg_sq, grad, dim,
                          beta1, beta2, eps, weight_scale, alpha);
        continue;
      }
#endif
      UpdateAdamRowScalar(embedding, exp_avg, exp_avg_sq, grad, dim,
                          beta1, beta2, eps, weight_scale, alpha);
    }
  }
  const auto math_end = clock::now();
  adam_beta1_t        = next_beta1_t;
  adam_beta2_t        = next_beta2_t;
  ++adam_step;

  if (stats != nullptr) {
    stats->engine_get_ns = static_cast<uint64_t>(
        std::chrono::duration_cast<std::chrono::nanoseconds>(
            get_end - get_start)
            .count());
    for (const auto& worker_stat : worker_stats) {
      stats->index_lookup_ns += worker_stat.index_lookup_ns;
      stats->index_lookup_max_ns =
          std::max(stats->index_lookup_max_ns, worker_stat.index_lookup_ns);
    }
    stats->slot_alloc_ns = static_cast<uint64_t>(
        std::chrono::duration_cast<std::chrono::nanoseconds>(
            alloc_end - alloc_start)
            .count());
    stats->math_ns = static_cast<uint64_t>(
        std::chrono::duration_cast<std::chrono::nanoseconds>(
            math_end - math_start)
            .count());
  }
  return true;
}

py::dict LocalIndexedKV::SgdUpdateIndexed(
    const py::array_t<int64_t, py::array::c_style | py::array::forcecast>& ids,
    const py::array_t<float, py::array::c_style | py::array::forcecast>& grads,
    double lr,
    bool release_gil) {
  using clock            = std::chrono::steady_clock;
  const auto total_start = clock::now();
  const auto keys        = CheckIdsView(ids);
  const auto convert_end = clock::now();
  auto g                 = grads.unchecked<2>();
  if (g.shape(0) != static_cast<py::ssize_t>(keys.size()) ||
      g.shape(1) != dim) {
    throw std::invalid_argument(
        "sgd_update_indexed: grads must be [num_ids, dim] float");
  }
  if (!(lr > 0.0)) {
    throw std::invalid_argument("sgd_update_indexed: lr must be positive");
  }

  const auto engine_start = clock::now();
  bool ok                 = true;
  {
    std::lock_guard<std::mutex> lock(mu);
    if (release_gil) {
      py::gil_scoped_release nogil;
      if (!keys.empty())
        ok = ApplySgdUpdateParallel(keys, g.data(0, 0), static_cast<float>(lr));
    } else if (!keys.empty()) {
      ok = ApplySgdUpdateParallel(keys, g.data(0, 0), static_cast<float>(lr));
    }
  }
  const auto engine_end = clock::now();
  if (!ok) {
    throw std::runtime_error(
        "ApplySgdUpdateFlatRange failed (all keys must already exist)");
  }
  auto ns = [](clock::duration d) {
    return static_cast<double>(
        std::chrono::duration_cast<std::chrono::nanoseconds>(d).count());
  };
  py::dict profile;
  profile["arg_convert_ns"]   = ns(convert_end - total_start);
  profile["engine_update_ns"] = ns(engine_end - engine_start);
  profile["total_ns"]         = ns(engine_end - total_start);
  profile["rows"]             = static_cast<double>(keys.size());
  profile["workers"] = static_cast<double>(EffectiveWorkers(keys.size()));
  return profile;
}

py::dict LocalIndexedKV::AdamWTFUpdateIndexed(
    const py::array_t<int64_t, py::array::c_style | py::array::forcecast>& ids,
    const py::array_t<float, py::array::c_style | py::array::forcecast>& grads,
    double lr,
    double beta1,
    double beta2,
    double eps,
    double weight_decay,
    bool release_gil) {
  using clock            = std::chrono::steady_clock;
  const auto total_start = clock::now();
  const auto keys        = CheckIdsView(ids);
  const auto convert_end = clock::now();
  auto g                 = grads.unchecked<2>();
  if (g.shape(0) != static_cast<py::ssize_t>(keys.size()) ||
      g.shape(1) != dim) {
    throw std::invalid_argument(
        "adamw_tf_update_indexed: grads must be [num_ids, dim] float");
  }
  if (!(beta1 >= 0.0 && beta1 < 1.0 && beta2 >= 0.0 && beta2 < 1.0) ||
      eps < 0.0 || lr < 0.0 || weight_decay < 0.0) {
    throw std::invalid_argument(
        "adamw_tf_update_indexed: invalid optimizer parameters");
  }

  const auto lock_start = clock::now();
  bool ok               = true;
  AdamUpdateStats update_stats;
  {
    std::lock_guard<std::mutex> lock(mu);
    const auto lock_end = clock::now();
    if (release_gil) {
      py::gil_scoped_release nogil;
      if (!keys.empty()) {
        ok = ApplyAdamWTFUpdateParallel(
            keys,
            g.data(0, 0),
            lr,
            beta1,
            beta2,
            eps,
            weight_decay,
            &update_stats);
      }
    } else if (!keys.empty()) {
      ok = ApplyAdamWTFUpdateParallel(
          keys,
          g.data(0, 0),
          lr,
          beta1,
          beta2,
          eps,
          weight_decay,
          &update_stats);
    }
    if (!ok) {
      throw std::runtime_error(
          "AdamWTF update failed (keys must exist and stable indices must "
          "be valid)");
    }
    const auto engine_end = clock::now();
    auto ns               = [](clock::duration d) {
      return static_cast<double>(
          std::chrono::duration_cast<std::chrono::nanoseconds>(d).count());
    };
    py::dict profile;
    profile["arg_convert_ns"]   = ns(convert_end - total_start);
    profile["lock_wait_ns"]     = ns(lock_end - lock_start);
    profile["engine_update_ns"] = ns(engine_end - lock_end);
    profile["engine_get_ns"] = static_cast<double>(update_stats.engine_get_ns);
    profile["engine_index_lookup_ns"] =
        static_cast<double>(update_stats.index_lookup_ns);
    profile["engine_index_lookup_max_ns"] =
        static_cast<double>(update_stats.index_lookup_max_ns);
    profile["slot_alloc_ns"] = static_cast<double>(update_stats.slot_alloc_ns);
    profile["adam_math_ns"]  = static_cast<double>(update_stats.math_ns);
    profile["adam_vectorized"] = adam_vectorized && dim >= 4 && AdamAvx2Available();
    profile["adam_prefetch_distance"] = adam_prefetch_distance;
    profile["total_ns"]      = ns(engine_end - total_start);
    profile["rows"]          = static_cast<double>(keys.size());
    profile["workers"] = static_cast<double>(EffectiveWorkers(keys.size()));
    profile["step"]    = static_cast<double>(adam_step);
    return profile;
  }
}

py::dict LocalIndexedKV::AdamWTFUpdateIndexedTwoBatches(
    const py::array_t<int64_t, py::array::c_style | py::array::forcecast>&
        ids_a,
    const py::array_t<float, py::array::c_style | py::array::forcecast>& grads_a,
    const py::array_t<int64_t, py::array::c_style | py::array::forcecast>&
        ids_b,
    const py::array_t<float, py::array::c_style | py::array::forcecast>& grads_b,
    double lr,
    double beta1,
    double beta2,
    double eps,
    double weight_decay,
    bool release_gil) {
  using clock            = std::chrono::steady_clock;
  const auto total_start = clock::now();
  const auto keys_a      = CheckIdsView(ids_a);
  const auto keys_b      = CheckIdsView(ids_b);
  auto g_a               = grads_a.unchecked<2>();
  auto g_b               = grads_b.unchecked<2>();
  if (g_a.shape(0) != static_cast<py::ssize_t>(keys_a.size()) ||
      g_b.shape(0) != static_cast<py::ssize_t>(keys_b.size()) ||
      g_a.shape(1) != dim || g_b.shape(1) != dim) {
    throw std::invalid_argument(
        "adamw_tf_update_indexed_two_batches: each gradient must be "
        "[num_ids, dim] float");
  }
  if (!(beta1 >= 0.0 && beta1 < 1.0 && beta2 >= 0.0 && beta2 < 1.0) ||
      eps < 0.0 || lr < 0.0 || weight_decay < 0.0) {
    throw std::invalid_argument(
        "adamw_tf_update_indexed_two_batches: invalid optimizer parameters");
  }
  const auto convert_end = clock::now();

  const auto lock_start = clock::now();
  std::lock_guard<std::mutex> lock(mu);
  const auto lock_end = clock::now();
  bool ordered_merge = false;
  bool ok            = true;
  AdamUpdateStats update_stats;
  clock::time_point merge_end;
  clock::time_point engine_end;
  auto execute = [&]() {
    ordered_merge = MergeTwoGradientBatches(
        keys_a,
        grads_a.data(),
        keys_b,
        grads_b.data(),
        static_cast<size_t>(dim),
        &merged_keys_scratch,
        &merged_grads_scratch);
    merge_end = clock::now();
    if (!merged_keys_scratch.empty()) {
      ok = ApplyAdamWTFUpdateParallel(
          Int64KeyView{merged_keys_scratch.data(), merged_keys_scratch.size()},
          merged_grads_scratch.data(),
          lr,
          beta1,
          beta2,
          eps,
          weight_decay,
          &update_stats);
    }
    engine_end = clock::now();
  };
  if (release_gil) {
    py::gil_scoped_release nogil;
    execute();
  } else {
    execute();
  }
  if (!ok) {
    throw std::runtime_error(
        "AdamWTF two-batch update failed (keys must already exist and stable "
        "indices must be valid)");
  }

  auto ns = [](clock::duration duration) {
    return static_cast<double>(
        std::chrono::duration_cast<std::chrono::nanoseconds>(duration).count());
  };
  py::dict profile;
  profile["arg_convert_ns"]   = ns(convert_end - total_start);
  profile["lock_wait_ns"]     = ns(lock_end - lock_start);
  profile["coalesce_ns"]      = ns(merge_end - lock_end);
  profile["engine_update_ns"] = ns(engine_end - merge_end);
  profile["engine_get_ns"] = static_cast<double>(update_stats.engine_get_ns);
  profile["engine_index_lookup_ns"] =
      static_cast<double>(update_stats.index_lookup_ns);
  profile["engine_index_lookup_max_ns"] =
      static_cast<double>(update_stats.index_lookup_max_ns);
  profile["slot_alloc_ns"] = static_cast<double>(update_stats.slot_alloc_ns);
  profile["adam_math_ns"]  = static_cast<double>(update_stats.math_ns);
  profile["total_ns"]      = ns(engine_end - total_start);
  profile["input_rows"] = static_cast<double>(keys_a.size() + keys_b.size());
  profile["rows"]       = static_cast<double>(merged_keys_scratch.size());
  profile["duplicate_rows"] = static_cast<double>(
      keys_a.size() + keys_b.size() - merged_keys_scratch.size());
  profile["ordered_merge"]          = ordered_merge;
  profile["adam_vectorized"] =
      adam_vectorized && dim >= 4 && AdamAvx2Available();
  profile["adam_prefetch_distance"] = adam_prefetch_distance;
  profile["workers"] = static_cast<double>(
      EffectiveWorkers(merged_keys_scratch.size()));
  profile["step"] = static_cast<double>(adam_step);
  return profile;
}

py::dict LocalIndexedKV::OptimizerStateStats() const {
  py::dict stats;
  stats["adam_step"]        = static_cast<double>(adam_step);
  stats["adam_beta1_t"]     = static_cast<double>(adam_beta1_t);
  stats["adam_beta2_t"]     = static_cast<double>(adam_beta2_t);
  stats["adam_slot_blocks"] = static_cast<double>(adam_slot_blocks.size());
  stats["adam_slots_eager"]  = eager_adam_slots;
  stats["adam_slot_rows"]   = static_cast<double>(
      adam_slot_blocks.size() * static_cast<size_t>(kAdamSlotBlockRows));
  stats["adam_slot_bytes"] = static_cast<double>(
      adam_slot_blocks.size() * static_cast<size_t>(kAdamSlotBlockRows) *
      static_cast<size_t>(dim) * sizeof(float) * 2);
  return stats;
}

} // namespace recstore::python
