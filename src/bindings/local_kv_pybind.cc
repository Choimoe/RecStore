// In-process KVEngine Python binding (no PS server, no RPC).
//
// Exposes RecStore's storage engine (KVEngineComposite: index + value store)
// directly to Python as `_local_kv.LocalKV` so clients like QuantaRec's
// DynamicEmbedding can use RecStore as a plain in-process KV backend.
//
// Mirrors the engine setup proven by benchmark_local_training.cc:
//   - PMMmapRegisterCenter DRAM backend
//   - KVEngineComposite with configurable index type + DRAM_VALUE_STORE
//   - ConcurrentSlabMemoryPool-backed DRAM value allocator
//
// Deliberately avoids a libtorch dependency: this .so must be LD_PRELOADed
// (the whole-archive engine carries initial-exec TLS too large for glibc's
// dlopen static-TLS surplus), and preloading a libtorch chain would interpose a
// foreign torch over the host process's torch. The compatibility API uses
// numpy, while the fast lookup-into API accepts validated CPU tensor pointers
// and writes directly into caller-owned buffers without linking libtorch.

#include <algorithm>
#include <array>
#include <atomic>
#include <chrono>
#include <cstdint>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <cmath>
#include <limits>
#include <memory>
#include <mutex>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <unistd.h>

#include "base/factory.h"
#include "base/json.h"
#include "memory/shm_file.h"
#include "storage/kv_engine/base_kv.h"
#include "storage/kv_engine/engine_selector.h"
#include "storage/index/utils/hash.h"

namespace py = pybind11;

namespace {

constexpr std::array<char, 8> kIndexedManifestMagic = {
    'R', 'S', 'I', 'D', 'X', 'M', '0', '1'};
constexpr std::array<char, 8> kIndexedStateMagic = {
    'R', 'S', 'I', 'D', 'X', 'S', '0', '1'};
constexpr uint32_t kIndexedCheckpointVersion = 1;
constexpr uint64_t kIndexedCheckpointChecksumSeed = 0x6a09e667f3bcc909ULL;

bool CheckedAdd(uint64_t left, uint64_t right, uint64_t* result) {
  if (result == nullptr ||
      left > (std::numeric_limits<uint64_t>::max)() - right) {
    return false;
  }
  *result = left + right;
  return true;
}

bool CheckedMultiply(uint64_t left, uint64_t right, uint64_t* result) {
  if (result == nullptr ||
      (right != 0 && left > (std::numeric_limits<uint64_t>::max)() / right)) {
    return false;
  }
  *result = left * right;
  return true;
}

bool IsSafeSidecarFilename(const std::string& filename) {
  if (filename.empty() || filename == "." || filename == "..") {
    return false;
  }
  const std::filesystem::path path(filename);
  return !path.is_absolute() && !path.has_parent_path() &&
         path.filename() == path;
}

bool WriteCheckpointBytes(std::ofstream& output,
                          const void* data,
                          size_t size) {
  if (size >
      static_cast<size_t>((std::numeric_limits<std::streamsize>::max)())) {
    return false;
  }
  if (size != 0) {
    output.write(static_cast<const char*>(data),
                 static_cast<std::streamsize>(size));
  }
  return output.good();
}

template <typename T>
bool WriteCheckpointPod(std::ofstream& output, const T& value) {
  return WriteCheckpointBytes(output, &value, sizeof(value));
}

bool ReadCheckpointBytes(std::ifstream& input,
                         void* data,
                         uint64_t size,
                         uint64_t* remaining) {
  if (remaining == nullptr || size > *remaining ||
      size > static_cast<uint64_t>(
                 (std::numeric_limits<std::streamsize>::max)())) {
    return false;
  }
  if (size != 0) {
    input.read(static_cast<char*>(data), static_cast<std::streamsize>(size));
    if (!input) {
      return false;
    }
  }
  *remaining -= size;
  return true;
}

template <typename T>
bool ReadCheckpointPod(std::ifstream& input,
                       T* value,
                       uint64_t* remaining) {
  return ReadCheckpointBytes(input, value, sizeof(*value), remaining);
}

void UpdateCheckpointChecksum(uint64_t* checksum,
                              const void* data,
                              size_t size) {
  if (size != 0) {
    *checksum = xxhash(data, size, *checksum);
  }
}

template <typename T>
void UpdateCheckpointChecksumPod(uint64_t* checksum, const T& value) {
  UpdateCheckpointChecksum(checksum, &value, sizeof(value));
}

struct IndexedCheckpointManifest {
  std::string metadata;
  std::string kv_filename;
  std::string state_filename;
};

IndexedCheckpointManifest ReadIndexedCheckpointManifest(
    const std::filesystem::path& path) {
  std::error_code error;
  const std::uintmax_t file_size = std::filesystem::file_size(path, error);
  if (error || file_size > std::numeric_limits<uint64_t>::max()) {
    throw std::runtime_error("failed to inspect RecStore checkpoint manifest");
  }
  uint64_t remaining = static_cast<uint64_t>(file_size);
  std::ifstream input(path, std::ios::binary);
  std::array<char, kIndexedManifestMagic.size()> magic{};
  uint32_t version = 0;
  uint64_t metadata_size = 0;
  uint64_t kv_name_size = 0;
  uint64_t state_name_size = 0;
  uint64_t payload_size = 0;
  uint64_t payload_and_checksum = 0;
  if (!input ||
      !ReadCheckpointBytes(input, magic.data(), magic.size(), &remaining) ||
      !ReadCheckpointPod(input, &version, &remaining) ||
      !ReadCheckpointPod(input, &metadata_size, &remaining) ||
      !ReadCheckpointPod(input, &kv_name_size, &remaining) ||
      !ReadCheckpointPod(input, &state_name_size, &remaining) ||
      magic != kIndexedManifestMagic ||
      version != kIndexedCheckpointVersion || metadata_size > remaining ||
      kv_name_size > remaining - metadata_size ||
      state_name_size > remaining - metadata_size - kv_name_size ||
      metadata_size > std::numeric_limits<size_t>::max() ||
      kv_name_size > std::numeric_limits<size_t>::max() ||
      state_name_size > std::numeric_limits<size_t>::max() ||
      !CheckedAdd(metadata_size, kv_name_size, &payload_size) ||
      !CheckedAdd(payload_size, state_name_size, &payload_size) ||
      !CheckedAdd(payload_size, sizeof(uint64_t), &payload_and_checksum) ||
      payload_and_checksum != remaining) {
    throw std::runtime_error("invalid RecStore checkpoint manifest header");
  }

  IndexedCheckpointManifest manifest;
  manifest.metadata.resize(static_cast<size_t>(metadata_size));
  manifest.kv_filename.resize(static_cast<size_t>(kv_name_size));
  manifest.state_filename.resize(static_cast<size_t>(state_name_size));
  if (!ReadCheckpointBytes(input, manifest.metadata.data(), metadata_size,
                           &remaining) ||
      !ReadCheckpointBytes(input, manifest.kv_filename.data(), kv_name_size,
                           &remaining) ||
      !ReadCheckpointBytes(input, manifest.state_filename.data(),
                           state_name_size, &remaining)) {
    throw std::runtime_error("truncated RecStore checkpoint manifest");
  }
  uint64_t saved_checksum = 0;
  if (!ReadCheckpointPod(input, &saved_checksum, &remaining) || remaining != 0) {
    throw std::runtime_error("invalid RecStore checkpoint manifest trailer");
  }
  uint64_t checksum = kIndexedCheckpointChecksumSeed;
  UpdateCheckpointChecksum(&checksum, magic.data(), magic.size());
  UpdateCheckpointChecksumPod(&checksum, version);
  UpdateCheckpointChecksumPod(&checksum, metadata_size);
  UpdateCheckpointChecksumPod(&checksum, kv_name_size);
  UpdateCheckpointChecksumPod(&checksum, state_name_size);
  UpdateCheckpointChecksum(&checksum, manifest.metadata.data(), metadata_size);
  UpdateCheckpointChecksum(&checksum, manifest.kv_filename.data(),
                           kv_name_size);
  UpdateCheckpointChecksum(&checksum, manifest.state_filename.data(),
                           state_name_size);
  if (saved_checksum != checksum ||
      !IsSafeSidecarFilename(manifest.kv_filename) ||
      !IsSafeSidecarFilename(manifest.state_filename)) {
    throw std::runtime_error("invalid RecStore checkpoint manifest");
  }
  return manifest;
}

}  // namespace

// Checked conversion used by the general LocalKV API.
std::vector<uint64_t> CheckIds(const py::array_t<int64_t, py::array::c_style |
                                             py::array::forcecast>& ids) {
  auto info = ids.unchecked<1>();
  std::vector<uint64_t> keys(info.shape(0));
  for (py::ssize_t i = 0; i < info.shape(0); ++i) {
    keys[i] = static_cast<uint64_t>(info(i));
  }
  return keys;
}

struct Int64KeyView {
  const uint64_t* values;
  size_t count;

  const uint64_t* data() const { return values; }
  size_t size() const { return count; }
  bool empty() const { return count == 0; }
  uint64_t operator[](size_t index) const { return values[index]; }
  base::ConstArray<uint64_t> AsConstArray() const {
    if (count > static_cast<size_t>(std::numeric_limits<int>::max())) {
      throw std::invalid_argument("ID batch exceeds RecStore array limit");
    }
    return base::ConstArray<uint64_t>(values, static_cast<int>(count));
  }
};

Int64KeyView CheckIdsView(
    const py::array_t<int64_t, py::array::c_style |
                                  py::array::forcecast>& ids) {
  auto info = ids.unchecked<1>();
  // RecStore treats IDs as opaque 64-bit keys. Reusing the contiguous int64
  // representation preserves the exact modulo-2^64 mapping used by CheckIds.
  return Int64KeyView{
      reinterpret_cast<const uint64_t*>(info.data(0)),
      static_cast<size_t>(info.shape(0))};
}

std::unique_ptr<BaseKV> MakeLocalKV(int64_t capacity, int64_t row_bytes,
                                    const std::string& index_type,
                                    const std::string& value_dir,
                                    const std::string& allocator,
                                    int64_t allocator_capacity,
                                    int num_threads, int index_prefetch = 1) {
  if (capacity <= 0 || row_bytes <= 0) {
    throw std::invalid_argument("capacity and row_bytes must be positive");
  }
  if (num_threads <= 0) num_threads = 1;
  std::filesystem::create_directories(value_dir);
  const uint64_t alloc_bytes =
      allocator_capacity > 0
          ? static_cast<uint64_t>(allocator_capacity)
          : std::max<uint64_t>(
                64ULL << 20,
                static_cast<uint64_t>(capacity) *
                    static_cast<uint64_t>(row_bytes + 8) * 2);

  base::PMMmapRegisterCenter::GetConfig().backend =
      base::PMMmapRegisterCenter::BackendFromUseDram(true);
  BaseKVConfig cfg;
  cfg.num_threads_ = static_cast<unsigned>(num_threads);
  cfg.json_config_ = {
      {"engine_type", "KVEngineComposite"},
      {"capacity", static_cast<uint64_t>(capacity)},
      {"index", {{"type", index_type}, {"prefetch_distance", index_prefetch}}},
      {"value",
       {{"type", "DRAM_VALUE_STORE"},
        {"path", value_dir},
        {"default_value_size_hint", row_bytes},
        {"dram_allocator", {{"type", allocator},
                            {"capacity_bytes", alloc_bytes}}}}}};
  auto resolved = base::ResolveEngine(cfg);
  std::unique_ptr<BaseKV> kv(
      base::Factory<BaseKV, const BaseKVConfig&>::NewInstance(
          resolved.engine, resolved.cfg));
  if (!kv) throw std::runtime_error("failed to create KVEngineComposite");
  return kv;
}

struct LocalKV {
  std::unique_ptr<BaseKV> kv;
  int64_t dim;
  std::string value_path;
  uint64_t adam_step = 0;
  float adam_beta1_t = 1.0f;
  float adam_beta2_t = 1.0f;
  std::unordered_map<uint64_t, std::vector<float>> adam_m;
  std::unordered_map<uint64_t, std::vector<float>> adam_v;

  LocalKV(int64_t capacity, int64_t embedding_dim, const std::string& index_type,
          const std::string& value_dir, const std::string& allocator,
          int64_t allocator_capacity, int num_threads)
      : dim(embedding_dim), value_path(value_dir) {
    if (capacity <= 0 || embedding_dim <= 0) {
      throw std::invalid_argument("capacity and embedding_dim must be positive");
    }
    kv = MakeLocalKV(capacity, embedding_dim * sizeof(float), index_type,
                     value_dir, allocator, allocator_capacity, num_threads);
  }

  // Returns [num_ids, dim] float32 numpy array; missing keys read as zeros.
  py::array_t<float> Lookup(const py::array_t<int64_t, py::array::c_style |
                                           py::array::forcecast>& ids) {
    const auto keys = CheckIds(ids);
    py::array_t<float> out({static_cast<py::ssize_t>(keys.size()),
                            static_cast<py::ssize_t>(dim)});
    float* out_ptr = out.mutable_data();
    if (keys.size()) {
      std::memset(out_ptr, 0, keys.size() * dim * sizeof(float));
      if (!kv->BatchGetFlat(base::ConstArray<uint64_t>(keys), out_ptr,
                            static_cast<int64_t>(keys.size()), dim, 0)) {
        throw std::runtime_error("BatchGetFlat failed");
      }
    }
    return out;
  }

  // Phase-profiled lookup: same work as Lookup but returns per-phase
  // nanoseconds so callers can attribute cost to each stage of the read path:
  //   arg_convert  numpy -> std::vector<uint64_t> (per-element cast loop)
  //   output_alloc py::array_t allocation + pre-memset
  //   index_lookup PetHash probe per key (HintPrefetch + GetInternal)
  //   row_copy     value store ReadFlatFixedRows (ptr translate + memcpy)
  //   zero_fill    memset of missing rows (0 when all hit)
  //   total        whole call
  std::pair<py::array_t<float>, py::dict> LookupProfiled(
      const py::array_t<int64_t, py::array::c_style | py::array::forcecast>& ids) {
    using clock = std::chrono::steady_clock;
    const auto t0 = clock::now();
    const auto keys = CheckIds(ids);
    const auto t1  = clock::now();
    py::array_t<float> out({static_cast<py::ssize_t>(keys.size()),
                            static_cast<py::ssize_t>(dim)});
    float* out_ptr = out.mutable_data();
    std::memset(out_ptr, 0, keys.size() * dim * sizeof(float));
    const auto t2        = clock::now();
    BaseKV::BatchGetFlatStats stats;
    bool ok = true;
    if (keys.size()) {
      ok = kv->BatchGetFlat(base::ConstArray<uint64_t>(keys), out_ptr,
                            static_cast<int64_t>(keys.size()), dim, 0, &stats);
    }
    const auto t3 = clock::now();
    if (!ok) throw std::runtime_error("BatchGetFlat failed");
    auto ns = [](clock::duration d) {
      return static_cast<double>(
          std::chrono::duration_cast<std::chrono::nanoseconds>(d).count());
    };
    py::dict ph;
    ph["arg_convert_ns"]  = ns(t1 - t0);
    ph["output_alloc_ns"] = ns(t2 - t1);
    ph["index_lookup_ns"] = static_cast<double>(stats.index_lookup_ns);
    ph["row_copy_ns"]     = static_cast<double>(stats.row_copy_ns);
    ph["zero_fill_ns"]    = static_cast<double>(stats.zero_fill_ns);
    ph["missing_rows"]    = static_cast<double>(stats.missing_rows);
    ph["total_ns"]        = ns(t3 - t0);
    return {std::move(out), std::move(ph)};
  }

  // Index probe only (no value copy): isolates PetHash lookup cost.
  py::dict IndexOnly(
      const py::array_t<int64_t, py::array::c_style | py::array::forcecast>& ids) {
    const auto keys = CheckIds(ids);
    BaseKV::BatchGetFlatStats stats;
    if (keys.size() &&
        !kv->BatchGetIndexOnly(base::ConstArray<uint64_t>(keys), 0, &stats)) {
      throw std::runtime_error("BatchGetIndexOnly failed");
    }
    py::dict ph;
    ph["missing_rows"] = static_cast<double>(stats.missing_rows);
    ph["num_keys"]     = static_cast<double>(keys.size());
    return ph;
  }

  // Lookup into a caller-provided [num_ids, dim] float32 buffer.
  // Lets the caller reuse one buffer across calls, skipping both the
  // per-call allocation and the pre-memset (ReadFlatFixedRows fills every
  // row, zero-filling misses itself).
  void LookupInto(const py::array_t<int64_t, py::array::c_style |
                                          py::array::forcecast>& ids,
                  py::array_t<float, py::array::c_style | py::array::forcecast> out) {
    const auto keys = CheckIds(ids);
    auto v = out.unchecked<2>();
    if (v.shape(0) != static_cast<py::ssize_t>(keys.size()) ||
        v.shape(1) != static_cast<py::ssize_t>(dim)) {
      throw std::invalid_argument("lookup_into: out must be [num_ids, dim] float32");
    }
    float* out_ptr = out.mutable_data();
    if (keys.size()) {
      // The engine call touches no Python objects: release the GIL so
      // concurrent lookups from multiple Python threads actually overlap.
      py::gil_scoped_release nogil;
      if (!kv->BatchGetFlat(base::ConstArray<uint64_t>(keys), out_ptr,
                            static_cast<int64_t>(keys.size()), dim, 0)) {
        throw std::runtime_error("BatchGetFlat failed");
      }
    }
  }

  void Put(const py::array_t<int64_t, py::array::c_style | py::array::forcecast>& ids,
           const py::array_t<float, py::array::c_style | py::array::forcecast>& values) {
    const auto keys = CheckIds(ids);
    auto vals = values.unchecked<2>();
    if (vals.shape(0) != static_cast<py::ssize_t>(keys.size()) ||
        vals.shape(1) != dim) {
      throw std::invalid_argument("put: values must be [num_ids, dim] float");
    }
    const auto* data = vals.data(0, 0);
    std::vector<base::ConstArray<float>> arrays(keys.size());
    for (size_t i = 0; i < keys.size(); ++i) {
      arrays[i] = base::ConstArray<float>(data + i * dim, dim);
    }
    {
      py::gil_scoped_release nogil;
      kv->BatchPut(base::ConstArray<uint64_t>(keys), &arrays, 0);
    }
  }

  // release_gil=false (serial path): holding the GIL through a sub-millisecond
  // engine call avoids a costly GIL reacquisition after release — in a busy
  // process (e.g. a dataloader thread) the caller can wait several ms behind
  // another thread's GIL slice. release_gil=true (parallel path): worker
  // threads need the release so their engine calls actually overlap.
  void SgdUpdate(const py::array_t<int64_t, py::array::c_style |
                                    py::array::forcecast>& ids,
                 const py::array_t<float, py::array::c_style |
                                    py::array::forcecast>& grads,
                 double lr,
                 bool release_gil) {
    const auto keys = CheckIds(ids);
    auto g = grads.unchecked<2>();
    if (g.shape(0) != static_cast<py::ssize_t>(keys.size()) || g.shape(1) != dim) {
      throw std::invalid_argument("sgd_update: grads must be [num_ids, dim] float");
    }
    if (release_gil) {
      py::gil_scoped_release nogil;
      if (!kv->ApplySgdUpdateFlat(
              base::ConstArray<uint64_t>(keys), g.data(0, 0),
              static_cast<int64_t>(keys.size()), dim, static_cast<float>(lr),
              0, 0)) {
        throw std::runtime_error("ApplySgdUpdateFlat failed");
      }
    } else if (!kv->ApplySgdUpdateFlat(
                   base::ConstArray<uint64_t>(keys), g.data(0, 0),
                   static_cast<int64_t>(keys.size()), dim,
                   static_cast<float>(lr), 0, 0)) {
      throw std::runtime_error("ApplySgdUpdateFlat failed");
    }
  }

  void AdamWUpdate(
      const py::array_t<int64_t, py::array::c_style | py::array::forcecast>& ids,
      const py::array_t<float, py::array::c_style | py::array::forcecast>& grads,
      double lr, double beta1, double beta2, double eps, double weight_decay,
      bool release_gil) {
    const auto keys = CheckIds(ids);
    auto g = grads.unchecked<2>();
    if (g.shape(0) != static_cast<py::ssize_t>(keys.size()) ||
        g.shape(1) != dim) {
      throw std::invalid_argument("adamw_update: grads must be [num_ids, dim] float");
    }
    if (!(beta1 >= 0.0 && beta1 < 1.0 && beta2 >= 0.0 && beta2 < 1.0) ||
        eps <= 0.0 || lr <= 0.0 || weight_decay < 0.0) {
      throw std::invalid_argument("adamw_update: invalid optimizer parameters");
    }

    const auto* grad_ptr = g.data(0, 0);
    const size_t n = keys.size();
    std::vector<float> rows(n * static_cast<size_t>(dim));
    std::vector<float> effective(n * static_cast<size_t>(dim));
    // release_gil=false: serial callers avoid the GIL reacquisition stall
    // (see SgdUpdate). release_gil=true: needed for overlap from worker
    // threads. The engine calls and the pure-C++ math below touch no Python
    // objects either way.
    std::unique_ptr<py::gil_scoped_release> nogil;
    if (release_gil) nogil = std::make_unique<py::gil_scoped_release>();
    if (n > 0 && !kv->BatchGetFlat(
                       base::ConstArray<uint64_t>(keys), rows.data(),
                       static_cast<int64_t>(n), dim, 0)) {
      throw std::runtime_error("BatchGetFlat failed");
    }

    ++adam_step;
    // Match SparseAdamWTF's float beta accumulators and alpha formula.
    adam_beta1_t *= static_cast<float>(beta1);
    adam_beta2_t *= static_cast<float>(beta2);
    const float alpha =
        -static_cast<float>(lr) / (1.0f - adam_beta1_t) *
        std::sqrt(1.0f - adam_beta2_t);
    const double one_minus_weight_decay = 1.0 - weight_decay;
    for (size_t i = 0; i < n; ++i) {
      auto& m = adam_m[keys[i]];
      auto& v = adam_v[keys[i]];
      if (m.empty()) {
        m.assign(static_cast<size_t>(dim), 0.0f);
        v.assign(static_cast<size_t>(dim), 0.0f);
      }
      const float* grad = grad_ptr + i * static_cast<size_t>(dim);
      float* out = effective.data() + i * static_cast<size_t>(dim);
      const float* row = rows.data() + i * static_cast<size_t>(dim);
      for (int64_t c = 0; c < dim; ++c) {
        const size_t index = static_cast<size_t>(c);
        const float grad_value = grad[index];
        m[index] = static_cast<float>(beta1 * m[index] +
                                       (1.0 - beta1) * grad_value);
        v[index] = static_cast<float>(beta2 * v[index] +
                                       (1.0 - beta2) * grad_value * grad_value);
        // ApplySgdUpdateFlat computes row -= lr * effective. Encode the
        // native TensorFlow-style AdamW result as that effective gradient:
        // row' = row * (1 - weight_decay) + alpha * m / (sqrt(v) + eps).
        float updated = static_cast<float>(
            one_minus_weight_decay * static_cast<double>(row[index]));
        const float sqrt_v = std::sqrt(v[index]);
        const double adam_term =
            static_cast<double>(alpha) * static_cast<double>(m[index]) /
            (static_cast<double>(sqrt_v) + eps);
        updated = static_cast<float>(static_cast<double>(updated) + adam_term);
        out[index] = static_cast<float>(
            (static_cast<double>(row[index]) - updated) / lr);
      }
    }

    if (n > 0 && !kv->ApplySgdUpdateFlat(
                       base::ConstArray<uint64_t>(keys), effective.data(),
                       static_cast<int64_t>(n), dim, static_cast<float>(lr),
                       0, 0)) {
      throw std::runtime_error("ApplySgdUpdateFlat failed");
    }
  }
};

// Embedding-oriented RecStore backend. Each value row is laid out as
// [int64 stable_row_index][float embedding...]. RecStore remains the only
// key index, while Quanta can use stable_row_index for its optimizer slots.
struct LocalIndexedKV {
  static constexpr int64_t kAdamSlotBlockRows = 65536;

  struct AdamSlotBlock {
    explicit AdamSlotBlock(int64_t dim)
        : exp_avg(static_cast<size_t>(kAdamSlotBlockRows) *
                      static_cast<size_t>(dim),
                  0.0f),
          exp_avg_sq(static_cast<size_t>(kAdamSlotBlockRows) *
                         static_cast<size_t>(dim),
                     0.0f) {}

    std::vector<float> exp_avg;
    std::vector<float> exp_avg_sq;
  };

  struct AdamUpdateStats {
    uint64_t index_lookup_ns = 0;
    uint64_t index_lookup_max_ns = 0;
    uint64_t engine_get_ns = 0;
    uint64_t slot_alloc_ns = 0;
    uint64_t math_ns = 0;
  };

  struct RestoredIndexedState {
    int64_t next_row_index = 0;
    uint64_t active_rows = 0;
    uint64_t adam_step = 0;
    float adam_beta1_t = 1.0f;
    float adam_beta2_t = 1.0f;
    int64_t slot_rows = 0;
    std::vector<int64_t> free_rows;
    std::vector<float> exp_avg;
    std::vector<float> exp_avg_sq;
  };

  std::unique_ptr<BaseKV> kv;
  int64_t capacity;
  int64_t dim;
  int64_t storage_dim;
  int num_threads;
  int64_t next_row_index = 0;
  std::vector<int64_t> free_row_indices;
  uint64_t adam_step = 0;
  float adam_beta1_t = 1.0f;
  float adam_beta2_t = 1.0f;
  std::vector<std::unique_ptr<AdamSlotBlock>> adam_slot_blocks;
  std::vector<std::vector<BaseKV::DirectFixedRow>> worker_rows_scratch;
  std::vector<int64_t> row_indices_scratch;
  std::mutex mu;

  LocalIndexedKV(int64_t capacity_arg, int64_t embedding_dim,
                 const std::string& index_type, const std::string& value_dir,
                 const std::string& allocator, int64_t allocator_capacity,
                 int num_threads, int index_prefetch)
      : capacity(capacity_arg),
        dim(embedding_dim),
        storage_dim(embedding_dim +
                    static_cast<int64_t>(sizeof(int64_t) / sizeof(float))),
        num_threads(std::max(1, num_threads)) {
    if (embedding_dim <= 0) {
      throw std::invalid_argument("embedding_dim must be positive");
    }
    kv = MakeLocalKV(capacity, storage_dim * sizeof(float), index_type,
                     value_dir, allocator, allocator_capacity, num_threads,
                     index_prefetch);
    worker_rows_scratch.resize(static_cast<size_t>(this->num_threads));
  }

  int EffectiveWorkers(size_t rows) const {
    constexpr size_t kMinRowsPerWorker = 4096;
    if (rows == 0) return 1;
    const size_t useful_workers =
        (rows + kMinRowsPerWorker - 1) / kMinRowsPerWorker;
    return std::max(
        1, std::min(num_threads, static_cast<int>(useful_workers)));
  }

  int64_t AllocateRowIndex() {
    if (!free_row_indices.empty()) {
      const int64_t row_index = free_row_indices.back();
      free_row_indices.pop_back();
      return row_index;
    }
    if (next_row_index >= capacity) {
      throw std::runtime_error("LocalIndexedKV row-index capacity exceeded");
    }
    return next_row_index++;
  }

  void ClearAdamSlotRow(int64_t row_index) {
    const size_t block =
        static_cast<size_t>(row_index / kAdamSlotBlockRows);
    if (block >= adam_slot_blocks.size()) {
      return;
    }
    std::memset(AdamSlotPtr(row_index, false), 0,
                static_cast<size_t>(dim) * sizeof(float));
    std::memset(AdamSlotPtr(row_index, true), 0,
                static_cast<size_t>(dim) * sizeof(float));
  }

  void EnsureAdamSlotRows(int64_t rows) {
    if (rows < 0 || rows > capacity) {
      throw std::runtime_error("Adam slot row count exceeds engine capacity");
    }
    const size_t required_blocks = static_cast<size_t>(
        (rows + kAdamSlotBlockRows - 1) / kAdamSlotBlockRows);
    while (adam_slot_blocks.size() < required_blocks) {
      adam_slot_blocks.push_back(std::make_unique<AdamSlotBlock>(dim));
    }
  }

  float* AdamSlotPtr(int64_t row_index, bool squared) {
    const size_t block =
        static_cast<size_t>(row_index / kAdamSlotBlockRows);
    const size_t row =
        static_cast<size_t>(row_index % kAdamSlotBlockRows);
    auto& values = squared ? adam_slot_blocks[block]->exp_avg_sq
                           : adam_slot_blocks[block]->exp_avg;
    return values.data() + row * static_cast<size_t>(dim);
  }

  bool ApplySgdUpdateParallel(const Int64KeyView& keys,
                              const float* grads,
                              float lr) {
    const int workers = EffectiveWorkers(keys.size());
    std::vector<uint8_t> worker_ok(static_cast<size_t>(workers), 1);
#pragma omp parallel for num_threads(workers) schedule(static)
    for (int worker = 0; worker < workers; ++worker) {
      const size_t begin = keys.size() * static_cast<size_t>(worker) /
                           static_cast<size_t>(workers);
      const size_t end = keys.size() * static_cast<size_t>(worker + 1) /
                         static_cast<size_t>(workers);
      if (begin == end) continue;
      worker_ok[static_cast<size_t>(worker)] =
          kv->ApplySgdUpdateFlatRange(
              base::ConstArray<uint64_t>(
                  keys.data() + begin, static_cast<int>(end - begin)),
              grads + begin * static_cast<size_t>(dim),
              static_cast<int64_t>(end - begin), storage_dim,
              static_cast<int64_t>(sizeof(int64_t) / sizeof(float)), dim, lr,
              static_cast<unsigned>(worker))
              ? 1
              : 0;
    }
    return std::all_of(worker_ok.begin(), worker_ok.end(),
                       [](uint8_t ok) { return ok != 0; });
  }

  bool ApplyAdamWTFUpdateParallel(const Int64KeyView& keys,
                                  const float* grads,
                                  double lr,
                                  double beta1,
                                  double beta2,
                                  double eps,
                                  double weight_decay,
                                  AdamUpdateStats* stats) {
    using clock = std::chrono::steady_clock;
    const int workers = EffectiveWorkers(keys.size());
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
      if (begin == end) continue;
      auto& local_rows = worker_rows_scratch[static_cast<size_t>(worker)];
      const bool ok = kv->BatchGetDirectFixedRows(
          base::ConstArray<uint64_t>(
              keys.data() + begin, static_cast<int>(end - begin)),
          static_cast<int64_t>(end - begin), storage_dim,
          static_cast<unsigned>(worker), &local_rows,
          &worker_stats[static_cast<size_t>(worker)]);
      if (!ok || local_rows.size() != end - begin) {
        worker_ok[static_cast<size_t>(worker)] = 0;
        continue;
      }
    }
    const auto get_end = clock::now();
    if (!std::all_of(worker_ok.begin(), worker_ok.end(),
                     [](uint8_t ok) { return ok != 0; })) {
      return false;
    }

    auto& row_indices = row_indices_scratch;
    row_indices.resize(keys.size());
    int invalid_row = 0;
    int64_t max_row_index = -1;
#pragma omp parallel for num_threads(workers) schedule(static) \
    reduction(| : invalid_row) reduction(max : max_row_index)
    for (int worker = 0; worker < workers; ++worker) {
      const size_t begin = keys.size() * static_cast<size_t>(worker) /
                           static_cast<size_t>(workers);
      const auto& local_rows =
          worker_rows_scratch[static_cast<size_t>(worker)];
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
        max_row_index = std::max(max_row_index, row_index);
      }
    }
    if (invalid_row != 0) return false;

    const auto alloc_start = clock::now();
    EnsureAdamSlotRows(max_row_index + 1);
    const auto alloc_end = clock::now();

    const float next_beta1_t =
        adam_beta1_t * static_cast<float>(beta1);
    const float next_beta2_t =
        adam_beta2_t * static_cast<float>(beta2);
    const float alpha =
        -static_cast<float>(lr) / (1.0f - next_beta1_t) *
        std::sqrt(1.0f - next_beta2_t);
    const double weight_scale = 1.0 - weight_decay;
    const auto math_start = clock::now();
#pragma omp parallel for num_threads(workers) schedule(static)
    for (int worker = 0; worker < workers; ++worker) {
      const size_t begin = keys.size() * static_cast<size_t>(worker) /
                           static_cast<size_t>(workers);
      const auto& local_rows =
          worker_rows_scratch[static_cast<size_t>(worker)];
      for (size_t local = 0; local < local_rows.size(); ++local) {
        const size_t row_pos = begin + local;
        float* embedding = reinterpret_cast<float*>(
            const_cast<char*>(local_rows[local].data) + sizeof(int64_t));
        float* exp_avg = AdamSlotPtr(row_indices[row_pos], false);
        float* exp_avg_sq = AdamSlotPtr(row_indices[row_pos], true);
        const float* grad = grads + row_pos * static_cast<size_t>(dim);
        for (int64_t c = 0; c < dim; ++c) {
          const size_t col = static_cast<size_t>(c);
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
              static_cast<double>(alpha) * static_cast<double>(exp_avg[col]) /
                  (static_cast<double>(std::sqrt(exp_avg_sq[col])) + eps));
        }
      }
    }
    const auto math_end = clock::now();
    adam_beta1_t = next_beta1_t;
    adam_beta2_t = next_beta2_t;
    ++adam_step;

    if (stats != nullptr) {
      stats->engine_get_ns = static_cast<uint64_t>(
          std::chrono::duration_cast<std::chrono::nanoseconds>(get_end -
                                                               get_start)
              .count());
      for (const auto& worker_stat : worker_stats) {
        stats->index_lookup_ns += worker_stat.index_lookup_ns;
        stats->index_lookup_max_ns =
            std::max(stats->index_lookup_max_ns, worker_stat.index_lookup_ns);
      }
      stats->slot_alloc_ns = static_cast<uint64_t>(
          std::chrono::duration_cast<std::chrono::nanoseconds>(alloc_end -
                                                               alloc_start)
              .count());
      stats->math_ns = static_cast<uint64_t>(
          std::chrono::duration_cast<std::chrono::nanoseconds>(math_end -
                                                               math_start)
              .count());
    }
    return true;
  }

  py::tuple LookupIndexedImpl(
      const py::array_t<int64_t, py::array::c_style | py::array::forcecast>& ids,
      bool allocate_missing,
      py::dict* profile) {
    using clock = std::chrono::steady_clock;
    const auto total_start = clock::now();
    const auto keys = CheckIdsView(ids);
    const auto convert_end = clock::now();
    const auto n = static_cast<py::ssize_t>(keys.size());
    py::array_t<int64_t> indices(n);
    py::array_t<float> values({n, static_cast<py::ssize_t>(dim)});
    py::array_t<bool> found(n);
    auto* index_ptr = indices.mutable_data();
    auto* value_ptr = values.mutable_data();
    auto* found_ptr = found.mutable_data();
    const auto alloc_end = clock::now();

    std::unique_lock<std::mutex> lock(mu);
    const auto lock_end = clock::now();
    const int workers = EffectiveWorkers(keys.size());
    std::vector<BaseKV::BatchGetFlatStats> worker_stats(
        static_cast<size_t>(workers));
    std::vector<uint8_t> worker_ok(static_cast<size_t>(workers), 1);
#pragma omp parallel for num_threads(workers) schedule(static)
    for (int worker = 0; worker < workers; ++worker) {
      const size_t begin = keys.size() * static_cast<size_t>(worker) /
                           static_cast<size_t>(workers);
      const size_t end = keys.size() * static_cast<size_t>(worker + 1) /
                         static_cast<size_t>(workers);
      if (begin == end) continue;
      auto& local_rows = worker_rows_scratch[static_cast<size_t>(worker)];
      const bool ok = kv->BatchGetDirectFixedRows(
          base::ConstArray<uint64_t>(
              keys.data() + begin, static_cast<int>(end - begin)),
          static_cast<int64_t>(end - begin), storage_dim,
          static_cast<unsigned>(worker), &local_rows,
          &worker_stats[static_cast<size_t>(worker)]);
      if (!ok || local_rows.size() != end - begin) {
        worker_ok[static_cast<size_t>(worker)] = 0;
        continue;
      }
    }
    if (!std::all_of(worker_ok.begin(), worker_ok.end(),
                     [](uint8_t ok) { return ok != 0; })) {
      throw std::runtime_error("BatchGetDirectFixedRows failed");
    }
    const auto engine_end = clock::now();
    std::unordered_map<uint64_t, int64_t> new_indices;
    const size_t row_bytes = static_cast<size_t>(storage_dim) * sizeof(float);
    int invalid_row = 0;
#pragma omp parallel for num_threads(workers) schedule(static) \
    reduction(| : invalid_row)
    for (int worker = 0; worker < workers; ++worker) {
      const size_t begin = keys.size() * static_cast<size_t>(worker) /
                           static_cast<size_t>(workers);
      const auto& local_rows =
          worker_rows_scratch[static_cast<size_t>(worker)];
      for (size_t local = 0; local < local_rows.size(); ++local) {
        const size_t i = begin + local;
        const auto& row = local_rows[local];
        if (row.missing) {
          found_ptr[i] = false;
          std::memset(value_ptr + i * static_cast<size_t>(dim), 0,
                      static_cast<size_t>(dim) * sizeof(float));
          index_ptr[i] = -1;
          continue;
        }
        if (row.data == nullptr || row.size != row_bytes) {
          invalid_row = 1;
          continue;
        }
        found_ptr[i] = true;
        std::memcpy(index_ptr + i, row.data, sizeof(int64_t));
        std::memcpy(value_ptr + i * static_cast<size_t>(dim),
                    row.data + sizeof(int64_t),
                    static_cast<size_t>(dim) * sizeof(float));
      }
    }
    if (invalid_row != 0) {
      throw std::runtime_error("LocalIndexedKV stored row has invalid size");
    }
    if (allocate_missing) {
      for (size_t i = 0; i < keys.size(); ++i) {
        if (found_ptr[i]) continue;
        auto it = new_indices.find(keys[i]);
        if (it == new_indices.end()) {
          it = new_indices.emplace(keys[i], AllocateRowIndex()).first;
        }
        index_ptr[i] = it->second;
      }
    }
    const auto unpack_end = clock::now();
    if (profile != nullptr) {
      auto ns = [](clock::duration d) {
        return static_cast<double>(
            std::chrono::duration_cast<std::chrono::nanoseconds>(d).count());
      };
      (*profile)["arg_convert_ns"] = ns(convert_end - total_start);
      (*profile)["output_alloc_ns"] = ns(alloc_end - convert_end);
      (*profile)["lock_wait_ns"] = ns(lock_end - alloc_end);
      (*profile)["engine_get_ns"] = ns(engine_end - lock_end);
      uint64_t index_lookup_ns = 0;
      uint64_t missing_rows = 0;
      for (const auto& stats : worker_stats) {
        index_lookup_ns += stats.index_lookup_ns;
        missing_rows += stats.missing_rows;
      }
      (*profile)["engine_index_lookup_ns"] =
          static_cast<double>(index_lookup_ns);
      (*profile)["unpack_copy_ns"] = ns(unpack_end - engine_end);
      (*profile)["total_ns"] = ns(unpack_end - total_start);
      (*profile)["missing_rows"] = static_cast<double>(missing_rows);
      (*profile)["rows"] = static_cast<double>(keys.size());
      (*profile)["workers"] = static_cast<double>(workers);
    }
    return py::make_tuple(std::move(indices), std::move(values),
                          std::move(found));
  }

  py::tuple LookupIndexed(
      const py::array_t<int64_t, py::array::c_style | py::array::forcecast>& ids,
      bool allocate_missing) {
    return LookupIndexedImpl(ids, allocate_missing, nullptr);
  }

  py::tuple LookupIndexedProfiled(
      const py::array_t<int64_t, py::array::c_style | py::array::forcecast>& ids,
      bool allocate_missing) {
    py::dict profile;
    py::tuple result = LookupIndexedImpl(ids, allocate_missing, &profile);
    return py::make_tuple(result[0], result[1], result[2], std::move(profile));
  }

  py::dict LookupIndexedIntoRaw(uintptr_t ids_address,
                                int64_t num_rows,
                                uintptr_t indices_address,
                                uintptr_t values_address,
                                uintptr_t found_address,
                                bool allocate_missing,
                                bool release_gil,
                                bool collect_profile) {
    return LookupIndexedIntoRawImpl(ids_address, num_rows, indices_address,
                                    values_address, found_address,
                                    allocate_missing, release_gil,
                                    collect_profile, true);
  }

  py::dict LookupIndexedLegacyIntoRaw(uintptr_t ids_address,
                                      int64_t num_rows,
                                      uintptr_t indices_address,
                                      uintptr_t values_address,
                                      uintptr_t found_address,
                                      bool allocate_missing,
                                      bool release_gil,
                                      bool collect_profile) {
    return LookupIndexedIntoRawImpl(ids_address, num_rows, indices_address,
                                    values_address, found_address,
                                    allocate_missing, release_gil,
                                    collect_profile, false);
  }

  py::dict LookupIndexedFusedIntoRaw(uintptr_t ids_address,
                                     int64_t num_rows,
                                     uintptr_t indices_address,
                                     uintptr_t values_address,
                                     uintptr_t found_address,
                                     bool allocate_missing,
                                     bool release_gil,
                                     bool collect_profile) {
    return LookupIndexedIntoRawImpl(ids_address, num_rows, indices_address,
                                    values_address, found_address,
                                    allocate_missing, release_gil,
                                    collect_profile, true);
  }

  py::dict LookupIndexedIntoRawImpl(uintptr_t ids_address,
                                    int64_t num_rows,
                                    uintptr_t indices_address,
                                    uintptr_t values_address,
                                    uintptr_t found_address,
                                    bool allocate_missing,
                                    bool release_gil,
                                    bool collect_profile,
                                    bool fused) {
    // This ABI-neutral entry point is intentionally internal. The Quanta
    // adapter validates dtype, device, contiguity, shape, and buffer lifetime
    // before passing the CPU tensor addresses used for this synchronous call.
    using clock = std::chrono::steady_clock;
    const auto total_start = clock::now();
    if (num_rows < 0 ||
        static_cast<uint64_t>(num_rows) >
            static_cast<uint64_t>((std::numeric_limits<int>::max)())) {
      throw std::invalid_argument(
          "lookup_indexed_into_raw: invalid row count");
    }
    if (num_rows > 0 &&
        (ids_address == 0 || indices_address == 0 || values_address == 0 ||
         found_address == 0)) {
      throw std::invalid_argument(
          "lookup_indexed_into_raw: non-empty tensors require valid pointers");
    }

    const auto keys = Int64KeyView{
        reinterpret_cast<const uint64_t*>(ids_address),
        static_cast<size_t>(num_rows)};
    auto* index_ptr = reinterpret_cast<int64_t*>(indices_address);
    auto* value_ptr = reinterpret_cast<float*>(values_address);
    auto* found_ptr = reinterpret_cast<bool*>(found_address);
    const auto convert_end = clock::now();

    const int workers = EffectiveWorkers(keys.size());
    std::vector<BaseKV::BatchGetFlatStats> worker_stats(
        static_cast<size_t>(workers));
    std::vector<uint8_t> worker_ok(static_cast<size_t>(workers), 1);
    std::vector<uint64_t> worker_get_ns(static_cast<size_t>(workers), 0);
    std::vector<uint64_t> worker_unpack_ns(static_cast<size_t>(workers), 0);
    clock::time_point lock_start;
    clock::time_point lock_end;
    clock::time_point engine_end;
    uint64_t missing_rows = 0;

    auto execute = [&]() {
      lock_start = clock::now();
      std::unique_lock<std::mutex> lock(mu);
      lock_end = clock::now();
      const size_t row_bytes =
          static_cast<size_t>(storage_dim) * sizeof(float);
#pragma omp parallel for num_threads(workers) schedule(static) \
    reduction(+ : missing_rows)
      for (int worker = 0; worker < workers; ++worker) {
        const size_t begin = keys.size() * static_cast<size_t>(worker) /
                             static_cast<size_t>(workers);
        const size_t end = keys.size() * static_cast<size_t>(worker + 1) /
                           static_cast<size_t>(workers);
        if (begin == end) continue;
        if (fused) {
          auto& stats = worker_stats[static_cast<size_t>(worker)];
          const auto get_start = clock::now();
          const bool ok = kv->BatchGetIndexedFlat(
              base::ConstArray<uint64_t>(
                  keys.data() + begin, static_cast<int>(end - begin)),
              index_ptr + begin, value_ptr + begin * static_cast<size_t>(dim),
              found_ptr + begin, static_cast<int64_t>(end - begin), dim,
              static_cast<unsigned>(worker), &stats, collect_profile);
          worker_get_ns[static_cast<size_t>(worker)] = static_cast<uint64_t>(
              std::chrono::duration_cast<std::chrono::nanoseconds>(
                  clock::now() - get_start).count());
          worker_ok[static_cast<size_t>(worker)] = ok ? 1 : 0;
          missing_rows += stats.missing_rows;
          continue;
        }
        auto& local_rows = worker_rows_scratch[static_cast<size_t>(worker)];
        const auto get_start = clock::now();
        const bool ok = kv->BatchGetDirectFixedRows(
            base::ConstArray<uint64_t>(
                keys.data() + begin, static_cast<int>(end - begin)),
            static_cast<int64_t>(end - begin), storage_dim,
            static_cast<unsigned>(worker), &local_rows,
            collect_profile ? &worker_stats[static_cast<size_t>(worker)]
                            : nullptr);
        const auto get_end = clock::now();
        worker_get_ns[static_cast<size_t>(worker)] =
            static_cast<uint64_t>(
                std::chrono::duration_cast<std::chrono::nanoseconds>(
                    get_end - get_start)
                    .count());
        if (!ok || local_rows.size() != end - begin) {
          worker_ok[static_cast<size_t>(worker)] = 0;
          continue;
        }

        const auto unpack_start = clock::now();
        for (size_t local = 0; local < local_rows.size(); ++local) {
          const size_t output_row = begin + local;
          const auto& row = local_rows[local];
          if (row.missing) {
            found_ptr[output_row] = false;
            std::memset(
                value_ptr + output_row * static_cast<size_t>(dim), 0,
                static_cast<size_t>(dim) * sizeof(float));
            index_ptr[output_row] = -1;
            ++missing_rows;
            continue;
          }
          if (row.data == nullptr || row.size != row_bytes) {
            worker_ok[static_cast<size_t>(worker)] = 0;
            break;
          }
          found_ptr[output_row] = true;
          std::memcpy(index_ptr + output_row, row.data, sizeof(int64_t));
          std::memcpy(
              value_ptr + output_row * static_cast<size_t>(dim),
              row.data + sizeof(int64_t),
              static_cast<size_t>(dim) * sizeof(float));
        }
        worker_unpack_ns[static_cast<size_t>(worker)] =
            static_cast<uint64_t>(
                std::chrono::duration_cast<std::chrono::nanoseconds>(
                    clock::now() - unpack_start)
                    .count());
      }
      if (!std::all_of(worker_ok.begin(), worker_ok.end(),
                       [](uint8_t ok) { return ok != 0; })) {
        throw std::runtime_error(
            "lookup_indexed_into_raw: engine lookup or row unpack failed");
      }

      if (allocate_missing && missing_rows != 0) {
        std::unordered_map<uint64_t, int64_t> new_indices;
        new_indices.reserve(static_cast<size_t>(missing_rows));
        for (size_t i = 0; i < keys.size(); ++i) {
          if (found_ptr[i]) continue;
          auto it = new_indices.find(keys[i]);
          if (it == new_indices.end()) {
            it = new_indices.emplace(keys[i], AllocateRowIndex()).first;
          }
          index_ptr[i] = it->second;
        }
      }
      engine_end = clock::now();
    };

    if (release_gil) {
      py::gil_scoped_release nogil;
      execute();
    } else {
      execute();
    }
    const auto total_end = clock::now();

    py::dict profile;
    profile["missing_rows"] = static_cast<double>(missing_rows);
    profile["rows"] = static_cast<double>(keys.size());
    profile["workers"] = static_cast<double>(workers);
    if (!collect_profile) return profile;

    uint64_t get_cpu_ns = 0;
    uint64_t get_max_ns = 0;
    uint64_t unpack_cpu_ns = 0;
    uint64_t unpack_max_ns = 0;
    uint64_t index_lookup_cpu_ns = 0;
    uint64_t index_lookup_max_ns = 0;
    for (int worker = 0; worker < workers; ++worker) {
      const size_t index = static_cast<size_t>(worker);
      get_cpu_ns += worker_get_ns[index];
      get_max_ns = std::max(get_max_ns, worker_get_ns[index]);
      unpack_cpu_ns += worker_unpack_ns[index];
      unpack_max_ns = std::max(unpack_max_ns, worker_unpack_ns[index]);
      index_lookup_cpu_ns += worker_stats[index].index_lookup_ns;
      index_lookup_max_ns =
          std::max(index_lookup_max_ns, worker_stats[index].index_lookup_ns);
    }
    auto ns = [](clock::duration duration) {
      return static_cast<double>(
          std::chrono::duration_cast<std::chrono::nanoseconds>(duration)
              .count());
    };
    profile["arg_convert_ns"] = ns(convert_end - total_start);
    profile["output_alloc_ns"] = 0.0;
    profile["lock_wait_ns"] = ns(lock_end - lock_start);
    // These are critical-path estimates inside the fused parallel region.
    profile["engine_get_ns"] = static_cast<double>(get_max_ns);
    profile["engine_get_cpu_ns"] = static_cast<double>(get_cpu_ns);
    // `engine_index_lookup_ns` is retained as the aggregate CPU time for
    // compatibility.  The max value is the corresponding critical-path
    // estimate; summing worker times must not be compared with wall time.
    profile["engine_index_lookup_ns"] =
        static_cast<double>(index_lookup_cpu_ns);
    profile["engine_index_lookup_cpu_ns"] =
        static_cast<double>(index_lookup_cpu_ns);
    profile["engine_index_lookup_max_ns"] =
        static_cast<double>(index_lookup_max_ns);
    profile["unpack_copy_ns"] = static_cast<double>(unpack_max_ns);
    profile["unpack_copy_cpu_ns"] = static_cast<double>(unpack_cpu_ns);
    if (fused) {
      uint64_t copy_cpu_ns = 0;
      uint64_t copy_max_ns = 0;
      for (const auto& stats : worker_stats) {
        copy_cpu_ns += stats.row_copy_ns;
        copy_max_ns = std::max(copy_max_ns, stats.row_copy_ns);
      }
      profile["row_copy_cpu_ns"] = static_cast<double>(copy_cpu_ns);
      profile["row_copy_max_ns"] = static_cast<double>(copy_max_ns);
    }
    profile["engine_get_unpack_ns"] = ns(engine_end - lock_end);
    profile["total_ns"] = ns(total_end - total_start);
    return profile;
  }

  uint64_t LookupValuesIntoRawImpl(uintptr_t ids_address,
                                   int64_t num_rows,
                                   uintptr_t values_address,
                                   bool release_gil,
                                   py::dict* profile,
                                   uintptr_t missing_positions_address = 0) {
    using clock = std::chrono::steady_clock;
    const bool collect_profile = profile != nullptr;
    const auto total_start = clock::now();
    if (num_rows < 0 ||
        static_cast<uint64_t>(num_rows) >
            static_cast<uint64_t>((std::numeric_limits<int>::max)())) {
      throw std::invalid_argument("lookup_values_into_raw: invalid row count");
    }
    if (num_rows > 0 && (ids_address == 0 || values_address == 0)) {
      throw std::invalid_argument(
          "lookup_values_into_raw: non-empty tensors require valid pointers");
    }

    const auto keys = Int64KeyView{
        reinterpret_cast<const uint64_t*>(ids_address),
        static_cast<size_t>(num_rows)};
    auto* value_ptr = reinterpret_cast<float*>(values_address);
    auto* missing_positions =
        reinterpret_cast<int64_t*>(missing_positions_address);
    const int workers = EffectiveWorkers(keys.size());
    std::vector<BaseKV::BatchGetFlatStats> worker_stats(
        static_cast<size_t>(workers));
    std::vector<uint8_t> worker_ok(static_cast<size_t>(workers), 1);
    clock::time_point lock_start;
    clock::time_point lock_end;
    clock::time_point engine_end;

    auto execute = [&]() {
      lock_start = clock::now();
      std::unique_lock<std::mutex> lock(mu);
      lock_end = clock::now();
#pragma omp parallel for num_threads(workers) schedule(static)
      for (int worker = 0; worker < workers; ++worker) {
        const size_t begin = keys.size() * static_cast<size_t>(worker) /
                             static_cast<size_t>(workers);
        const size_t end = keys.size() * static_cast<size_t>(worker + 1) /
                           static_cast<size_t>(workers);
        if (begin == end) continue;
        worker_ok[static_cast<size_t>(worker)] =
            kv->BatchGetFlatRange(
                base::ConstArray<uint64_t>(
                    keys.data() + begin, static_cast<int>(end - begin)),
                value_ptr + begin * static_cast<size_t>(dim),
                static_cast<int64_t>(end - begin), storage_dim,
                static_cast<int64_t>(sizeof(int64_t) / sizeof(float)), dim,
                static_cast<unsigned>(worker),
                &worker_stats[static_cast<size_t>(worker)], collect_profile,
                missing_positions == nullptr ? nullptr
                                             : missing_positions + begin)
                ? 1
                : 0;
      }
      engine_end = clock::now();
    };

    if (release_gil) {
      py::gil_scoped_release nogil;
      execute();
    } else {
      execute();
    }
    if (!std::all_of(worker_ok.begin(), worker_ok.end(),
                     [](uint8_t ok) { return ok != 0; })) {
      throw std::runtime_error("lookup_values_into_raw: engine lookup failed");
    }
    const auto compact_start = clock::now();
    if (missing_positions != nullptr) {
      size_t write = 0;
      for (int worker = 0; worker < workers; ++worker) {
        const size_t begin = keys.size() * static_cast<size_t>(worker) /
                             static_cast<size_t>(workers);
        const uint64_t count =
            worker_stats[static_cast<size_t>(worker)].missing_rows;
        for (uint64_t local = 0; local < count; ++local) {
          missing_positions[write++] =
              static_cast<int64_t>(begin) + missing_positions[begin + local];
        }
      }
    }
    const auto compact_end = clock::now();
    const auto total_end = clock::now();

    uint64_t missing_rows = 0;
    uint64_t index_lookup_cpu_ns = 0;
    uint64_t index_lookup_max_ns = 0;
    uint64_t row_copy_cpu_ns = 0;
    uint64_t row_copy_max_ns = 0;
    for (const auto& stats : worker_stats) {
      missing_rows += stats.missing_rows;
      index_lookup_cpu_ns += stats.index_lookup_ns;
      index_lookup_max_ns =
          std::max(index_lookup_max_ns, stats.index_lookup_ns);
      row_copy_cpu_ns += stats.row_copy_ns;
      row_copy_max_ns = std::max(row_copy_max_ns, stats.row_copy_ns);
    }

    if (!collect_profile) return missing_rows;

    auto ns = [](clock::duration duration) {
      return static_cast<double>(
          std::chrono::duration_cast<std::chrono::nanoseconds>(duration)
              .count());
    };
    (*profile)["missing_rows"] = static_cast<double>(missing_rows);
    (*profile)["rows"] = static_cast<double>(keys.size());
    (*profile)["workers"] = static_cast<double>(workers);
    (*profile)["lock_wait_ns"] = ns(lock_end - lock_start);
    (*profile)["engine_index_lookup_cpu_ns"] =
        static_cast<double>(index_lookup_cpu_ns);
    (*profile)["engine_index_lookup_max_ns"] =
        static_cast<double>(index_lookup_max_ns);
    (*profile)["row_copy_cpu_ns"] = static_cast<double>(row_copy_cpu_ns);
    (*profile)["row_copy_max_ns"] = static_cast<double>(row_copy_max_ns);
    (*profile)["engine_get_unpack_ns"] = ns(engine_end - lock_end);
    (*profile)["missing_positions_compact_ns"] =
        ns(compact_end - compact_start);
    (*profile)["total_ns"] = ns(total_end - total_start);
    return missing_rows;
  }

  uint64_t LookupValuesIntoRaw(uintptr_t ids_address,
                               int64_t num_rows,
                               uintptr_t values_address,
                               bool release_gil) {
    return LookupValuesIntoRawImpl(
        ids_address, num_rows, values_address, release_gil, nullptr);
  }

  uint64_t LookupValuesWithMissesIntoRaw(uintptr_t ids_address,
                                         int64_t num_rows,
                                         uintptr_t values_address,
                                         uintptr_t missing_positions_address,
                                         bool release_gil) {
    if (num_rows > 0 && missing_positions_address == 0) {
      throw std::invalid_argument(
          "lookup_values_with_misses_into_raw: missing positions require a "
          "valid pointer");
    }
    return LookupValuesIntoRawImpl(ids_address, num_rows, values_address,
                                   release_gil, nullptr,
                                   missing_positions_address);
  }

  py::dict LookupValuesIntoRawProfiled(uintptr_t ids_address,
                                       int64_t num_rows,
                                       uintptr_t values_address,
                                       bool release_gil) {
    py::dict profile;
    LookupValuesIntoRawImpl(
        ids_address, num_rows, values_address, release_gil, &profile);
    return profile;
  }

  py::dict LookupValuesWithMissesIntoRawProfiled(
      uintptr_t ids_address,
      int64_t num_rows,
      uintptr_t values_address,
      uintptr_t missing_positions_address,
      bool release_gil) {
    if (num_rows > 0 && missing_positions_address == 0) {
      throw std::invalid_argument(
          "lookup_values_with_misses_into_raw_profiled: missing positions "
          "require a valid pointer");
    }
    py::dict profile;
    LookupValuesIntoRawImpl(ids_address, num_rows, values_address, release_gil,
                            &profile, missing_positions_address);
    return profile;
  }

  int64_t Size() {
    std::lock_guard<std::mutex> lock(mu);
    const uint64_t size = kv->ActiveKeyCount();
    if (size > static_cast<uint64_t>((std::numeric_limits<int64_t>::max)())) {
      throw std::runtime_error("LocalIndexedKV active key count is too large");
    }
    return static_cast<int64_t>(size);
  }

  py::tuple SnapshotIndexed() {
    std::lock_guard<std::mutex> lock(mu);
    const std::vector<uint64_t> keys = kv->SnapshotKeys();
    if (keys.size() >
        static_cast<size_t>((std::numeric_limits<py::ssize_t>::max)())) {
      throw std::runtime_error("LocalIndexedKV snapshot exceeds numpy limit");
    }

    const auto n = static_cast<py::ssize_t>(keys.size());
    py::array_t<int64_t> ids(n);
    py::array_t<int64_t> indices(n);
    py::array_t<float> values({n, static_cast<py::ssize_t>(dim)});
    if (keys.empty()) {
      return py::make_tuple(std::move(ids), std::move(indices),
                            std::move(values));
    }
    static_assert(sizeof(uint64_t) == sizeof(int64_t));
    std::memcpy(ids.mutable_data(), keys.data(), keys.size() * sizeof(uint64_t));

    const int workers = EffectiveWorkers(keys.size());
    std::vector<uint8_t> worker_ok(static_cast<size_t>(workers), 1);
#pragma omp parallel for num_threads(workers) schedule(static)
    for (int worker = 0; worker < workers; ++worker) {
      const size_t begin = keys.size() * static_cast<size_t>(worker) /
                           static_cast<size_t>(workers);
      const size_t end = keys.size() * static_cast<size_t>(worker + 1) /
                         static_cast<size_t>(workers);
      if (begin == end) continue;
      auto& local_rows = worker_rows_scratch[static_cast<size_t>(worker)];
      const bool ok = kv->BatchGetDirectFixedRows(
          base::ConstArray<uint64_t>(
              keys.data() + begin, static_cast<int>(end - begin)),
          static_cast<int64_t>(end - begin), storage_dim,
          static_cast<unsigned>(worker), &local_rows);
      if (!ok || local_rows.size() != end - begin) {
        worker_ok[static_cast<size_t>(worker)] = 0;
        continue;
      }
    }
    if (!std::all_of(worker_ok.begin(), worker_ok.end(),
                     [](uint8_t ok) { return ok != 0; })) {
      throw std::runtime_error("LocalIndexedKV snapshot row lookup failed");
    }

    auto* index_ptr = indices.mutable_data();
    auto* value_ptr = values.mutable_data();
    const size_t row_bytes = static_cast<size_t>(storage_dim) * sizeof(float);
    int invalid_row = 0;
#pragma omp parallel for num_threads(workers) schedule(static) \
    reduction(| : invalid_row)
    for (int worker = 0; worker < workers; ++worker) {
      const size_t begin = keys.size() * static_cast<size_t>(worker) /
                           static_cast<size_t>(workers);
      const auto& local_rows =
          worker_rows_scratch[static_cast<size_t>(worker)];
      for (size_t local = 0; local < local_rows.size(); ++local) {
        const size_t output_row = begin + local;
        const auto& row = local_rows[local];
        if (row.missing || row.data == nullptr || row.size != row_bytes) {
          invalid_row = 1;
          continue;
        }
        std::memcpy(index_ptr + output_row, row.data, sizeof(int64_t));
        std::memcpy(value_ptr + output_row * static_cast<size_t>(dim),
                    row.data + sizeof(int64_t),
                    static_cast<size_t>(dim) * sizeof(float));
      }
    }
    if (invalid_row != 0) {
      throw std::runtime_error("LocalIndexedKV snapshot contains invalid rows");
    }
    return py::make_tuple(std::move(ids), std::move(indices),
                          std::move(values));
  }

  int64_t DeleteKeysLocked(const std::vector<uint64_t>& requested_keys,
                           bool reset_allocator) {
    std::vector<uint64_t> keys;
    keys.reserve(requested_keys.size());
    std::unordered_set<uint64_t> seen;
    seen.reserve(requested_keys.size());
    for (const uint64_t key : requested_keys) {
      if (seen.insert(key).second) {
        keys.push_back(key);
      }
    }

    std::vector<BaseKV::DirectFixedRow> rows;
    if (!keys.empty() &&
        !kv->BatchGetDirectFixedRows(
            base::ConstArray<uint64_t>(keys),
            static_cast<int64_t>(keys.size()), storage_dim, 0, &rows)) {
      throw std::runtime_error("LocalIndexedKV delete row lookup failed");
    }
    const size_t row_bytes = static_cast<size_t>(storage_dim) * sizeof(float);
    std::vector<int64_t> deleted_indices;
    deleted_indices.reserve(keys.size());
    for (const auto& row : rows) {
      if (row.missing) {
        continue;
      }
      if (row.data == nullptr || row.size != row_bytes) {
        throw std::runtime_error("LocalIndexedKV delete found an invalid row");
      }
      int64_t row_index = -1;
      std::memcpy(&row_index, row.data, sizeof(int64_t));
      if (row_index < 0 || row_index >= next_row_index) {
        throw std::runtime_error(
            "LocalIndexedKV delete found an invalid stable index");
      }
      deleted_indices.push_back(row_index);
    }

    const uint64_t deleted = keys.empty()
                                 ? 0
                                 : kv->BatchDelete(
                                       base::ConstArray<uint64_t>(keys), 0);
    if (deleted != deleted_indices.size()) {
      throw std::runtime_error(
          "LocalIndexedKV delete count does not match fetched rows");
    }
    for (const int64_t row_index : deleted_indices) {
      ClearAdamSlotRow(row_index);
    }
    if (reset_allocator) {
      if (kv->ActiveKeyCount() != 0) {
        throw std::runtime_error(
            "LocalIndexedKV clear left active keys in the engine");
      }
      next_row_index = 0;
      free_row_indices.clear();
    } else {
      free_row_indices.insert(free_row_indices.end(), deleted_indices.begin(),
                              deleted_indices.end());
    }
    return static_cast<int64_t>(deleted);
  }

  int64_t DeleteIndexed(
      const py::array_t<int64_t, py::array::c_style |
                                    py::array::forcecast>& ids) {
    const auto keys = CheckIds(ids);
    std::lock_guard<std::mutex> lock(mu);
    return DeleteKeysLocked(keys, false);
  }

  int64_t ClearIndexed() {
    std::lock_guard<std::mutex> lock(mu);
    return DeleteKeysLocked(kv->SnapshotKeys(), true);
  }

  std::vector<int64_t> NormalizedFreeRows() {
    const std::vector<uint64_t> keys = kv->SnapshotKeys();
    if (keys.size() != kv->ActiveKeyCount()) {
      throw std::runtime_error(
          "RecStore checkpoint key tracking/count mismatch");
    }
    std::vector<BaseKV::DirectFixedRow> rows;
    if (!keys.empty() &&
        !kv->BatchGetDirectFixedRows(
            base::ConstArray<uint64_t>(keys),
            static_cast<int64_t>(keys.size()), storage_dim, 0, &rows)) {
      throw std::runtime_error("RecStore checkpoint row lookup failed");
    }

    std::vector<uint8_t> occupied(static_cast<size_t>(next_row_index), 0);
    const size_t row_bytes = static_cast<size_t>(storage_dim) * sizeof(float);
    for (const auto& row : rows) {
      if (row.missing || row.data == nullptr || row.size != row_bytes) {
        throw std::runtime_error("RecStore checkpoint found an invalid row");
      }
      int64_t row_index = -1;
      std::memcpy(&row_index, row.data, sizeof(int64_t));
      if (row_index < 0 || row_index >= next_row_index ||
          occupied[static_cast<size_t>(row_index)] != 0) {
        throw std::runtime_error(
            "RecStore checkpoint found an invalid stable index");
      }
      occupied[static_cast<size_t>(row_index)] = 1;
    }

    std::vector<uint8_t> already_free(static_cast<size_t>(next_row_index), 0);
    for (const int64_t row_index : free_row_indices) {
      if (row_index < 0 || row_index >= next_row_index ||
          occupied[static_cast<size_t>(row_index)] != 0 ||
          already_free[static_cast<size_t>(row_index)] != 0) {
        throw std::runtime_error(
            "RecStore checkpoint found an invalid free stable index");
      }
      already_free[static_cast<size_t>(row_index)] = 1;
    }

    // Reclaim any index reserved by a failed insert while preserving the LIFO
    // order of explicitly deleted rows at the back of the free list.
    std::vector<int64_t> normalized;
    normalized.reserve(static_cast<size_t>(next_row_index) - keys.size());
    for (int64_t row_index = 0; row_index < next_row_index; ++row_index) {
      if (occupied[static_cast<size_t>(row_index)] == 0 &&
          already_free[static_cast<size_t>(row_index)] == 0) {
        normalized.push_back(row_index);
      }
    }
    normalized.insert(normalized.end(), free_row_indices.begin(),
                      free_row_indices.end());
    if (normalized.size() + keys.size() !=
        static_cast<size_t>(next_row_index)) {
      throw std::runtime_error(
          "RecStore checkpoint allocator state is inconsistent");
    }
    return normalized;
  }

  bool WriteIndexedState(const std::filesystem::path& state_path,
                         const std::string& metadata) {
    const std::vector<int64_t> normalized_free = NormalizedFreeRows();
    const uint64_t active_rows = kv->ActiveKeyCount();
    const int64_t slot_rows = adam_slot_blocks.empty()
                                  ? 0
                                  : std::min<int64_t>(
                                        next_row_index,
                                        static_cast<int64_t>(
                                            adam_slot_blocks.size()) *
                                            kAdamSlotBlockRows);
    const uint64_t metadata_size = static_cast<uint64_t>(metadata.size());
    const uint64_t free_count =
        static_cast<uint64_t>(normalized_free.size());
    std::ofstream output(state_path, std::ios::binary | std::ios::trunc);
    if (!output) {
      return false;
    }
    uint64_t checksum = kIndexedCheckpointChecksumSeed;
    auto write = [&](const void* data, size_t size) {
      UpdateCheckpointChecksum(&checksum, data, size);
      return WriteCheckpointBytes(output, data, size);
    };
    bool ok = write(kIndexedStateMagic.data(), kIndexedStateMagic.size()) &&
              write(&kIndexedCheckpointVersion,
                    sizeof(kIndexedCheckpointVersion)) &&
              write(&metadata_size, sizeof(metadata_size)) &&
              write(&capacity, sizeof(capacity)) && write(&dim, sizeof(dim)) &&
              write(&next_row_index, sizeof(next_row_index)) &&
              write(&active_rows, sizeof(active_rows)) &&
              write(&free_count, sizeof(free_count)) &&
              write(&adam_step, sizeof(adam_step)) &&
              write(&adam_beta1_t, sizeof(adam_beta1_t)) &&
              write(&adam_beta2_t, sizeof(adam_beta2_t)) &&
              write(&slot_rows, sizeof(slot_rows)) &&
              write(metadata.data(), metadata.size()) &&
              write(normalized_free.data(),
                    normalized_free.size() * sizeof(int64_t));

    for (bool squared : {false, true}) {
      int64_t remaining_rows = slot_rows;
      for (size_t block = 0;
           ok && remaining_rows > 0 && block < adam_slot_blocks.size();
           ++block) {
        const int64_t rows = std::min<int64_t>(remaining_rows,
                                               kAdamSlotBlockRows);
        const auto& values = squared ? adam_slot_blocks[block]->exp_avg_sq
                                     : adam_slot_blocks[block]->exp_avg;
        const size_t count =
            static_cast<size_t>(rows) * static_cast<size_t>(dim);
        ok = write(values.data(), count * sizeof(float));
        remaining_rows -= rows;
      }
      ok = ok && remaining_rows == 0;
    }
    ok = ok && WriteCheckpointPod(output, checksum);
    output.flush();
    ok = ok && output.good();
    output.close();
    return ok && !output.fail();
  }

  RestoredIndexedState ReadIndexedState(
      const std::filesystem::path& state_path,
      const std::string& expected_metadata) {
    std::error_code error;
    const std::uintmax_t file_size =
        std::filesystem::file_size(state_path, error);
    if (error || file_size > std::numeric_limits<uint64_t>::max()) {
      throw std::runtime_error("failed to inspect RecStore indexed state");
    }
    uint64_t remaining = static_cast<uint64_t>(file_size);
    std::ifstream input(state_path, std::ios::binary);
    if (!input) {
      throw std::runtime_error("failed to open RecStore indexed state");
    }
    uint64_t checksum = kIndexedCheckpointChecksumSeed;
    auto read = [&](void* data, uint64_t size) {
      const bool ok = ReadCheckpointBytes(input, data, size, &remaining);
      if (ok) {
        UpdateCheckpointChecksum(&checksum, data, static_cast<size_t>(size));
      }
      return ok;
    };

    std::array<char, kIndexedStateMagic.size()> magic{};
    uint32_t version = 0;
    uint64_t metadata_size = 0;
    int64_t saved_capacity = 0;
    int64_t saved_dim = 0;
    uint64_t free_count = 0;
    RestoredIndexedState state;
    if (!read(magic.data(), magic.size()) ||
        !read(&version, sizeof(version)) ||
        !read(&metadata_size, sizeof(metadata_size)) ||
        !read(&saved_capacity, sizeof(saved_capacity)) ||
        !read(&saved_dim, sizeof(saved_dim)) ||
        !read(&state.next_row_index, sizeof(state.next_row_index)) ||
        !read(&state.active_rows, sizeof(state.active_rows)) ||
        !read(&free_count, sizeof(free_count)) ||
        !read(&state.adam_step, sizeof(state.adam_step)) ||
        !read(&state.adam_beta1_t, sizeof(state.adam_beta1_t)) ||
        !read(&state.adam_beta2_t, sizeof(state.adam_beta2_t)) ||
        !read(&state.slot_rows, sizeof(state.slot_rows)) ||
        magic != kIndexedStateMagic ||
        version != kIndexedCheckpointVersion || saved_capacity != capacity ||
        saved_dim != dim || state.next_row_index < 0 ||
        state.next_row_index > capacity || state.slot_rows < 0 ||
        state.slot_rows > state.next_row_index ||
        state.active_rows > static_cast<uint64_t>(state.next_row_index) ||
        free_count > static_cast<uint64_t>(state.next_row_index) ||
        state.active_rows + free_count !=
            static_cast<uint64_t>(state.next_row_index) ||
        metadata_size > remaining ||
        !std::isfinite(state.adam_beta1_t) ||
        !std::isfinite(state.adam_beta2_t)) {
      throw std::runtime_error("invalid RecStore indexed state header");
    }
    if (state.slot_rows != 0 &&
        static_cast<uint64_t>(state.slot_rows) >
            std::numeric_limits<size_t>::max() /
                static_cast<uint64_t>(dim)) {
      throw std::runtime_error("RecStore indexed optimizer state is too large");
    }

    uint64_t free_bytes = 0;
    uint64_t slot_values_u64 = 0;
    uint64_t slot_bytes = 0;
    uint64_t required_bytes = 0;
    if (metadata_size > std::numeric_limits<size_t>::max() ||
        !CheckedMultiply(free_count, sizeof(int64_t), &free_bytes) ||
        !CheckedMultiply(static_cast<uint64_t>(state.slot_rows),
                         static_cast<uint64_t>(dim), &slot_values_u64) ||
        !CheckedMultiply(slot_values_u64, sizeof(float), &slot_bytes) ||
        free_bytes > std::numeric_limits<size_t>::max() ||
        slot_values_u64 > std::numeric_limits<size_t>::max() ||
        slot_bytes > std::numeric_limits<size_t>::max() ||
        !CheckedAdd(metadata_size, free_bytes, &required_bytes) ||
        !CheckedAdd(required_bytes, slot_bytes, &required_bytes) ||
        !CheckedAdd(required_bytes, slot_bytes, &required_bytes) ||
        !CheckedAdd(required_bytes, sizeof(uint64_t), &required_bytes) ||
        required_bytes != remaining) {
      throw std::runtime_error("invalid RecStore indexed state size");
    }

    std::string metadata(static_cast<size_t>(metadata_size), '\0');
    state.free_rows.resize(static_cast<size_t>(free_count));
    const size_t slot_values = static_cast<size_t>(slot_values_u64);
    state.exp_avg.resize(slot_values);
    state.exp_avg_sq.resize(slot_values);
    if (!read(metadata.data(), metadata_size) ||
        !read(state.free_rows.data(), free_bytes) ||
        !read(state.exp_avg.data(), slot_bytes) ||
        !read(state.exp_avg_sq.data(), slot_bytes) ||
        metadata != expected_metadata) {
      throw std::runtime_error("invalid RecStore indexed state payload");
    }
    uint64_t saved_checksum = 0;
    if (!ReadCheckpointPod(input, &saved_checksum, &remaining) ||
        remaining != 0 || saved_checksum != checksum) {
      throw std::runtime_error("invalid RecStore indexed state checksum");
    }

    std::vector<uint8_t> seen(static_cast<size_t>(state.next_row_index), 0);
    for (const int64_t row_index : state.free_rows) {
      if (row_index < 0 || row_index >= state.next_row_index ||
          seen[static_cast<size_t>(row_index)] != 0) {
        throw std::runtime_error("invalid RecStore checkpoint free list");
      }
      seen[static_cast<size_t>(row_index)] = 1;
    }
    return state;
  }

  void ValidateRestoredRows(const RestoredIndexedState& state) {
    if (kv->ActiveKeyCount() != state.active_rows) {
      throw std::runtime_error("RecStore restored active-row count mismatch");
    }
    const std::vector<uint64_t> keys = kv->SnapshotKeys();
    std::vector<BaseKV::DirectFixedRow> rows;
    if (!keys.empty() &&
        !kv->BatchGetDirectFixedRows(
            base::ConstArray<uint64_t>(keys),
            static_cast<int64_t>(keys.size()), storage_dim, 0, &rows)) {
      throw std::runtime_error("RecStore restored row lookup failed");
    }
    std::vector<uint8_t> seen(static_cast<size_t>(state.next_row_index), 0);
    for (const int64_t row_index : state.free_rows) {
      seen[static_cast<size_t>(row_index)] = 1;
    }
    const size_t row_bytes = static_cast<size_t>(storage_dim) * sizeof(float);
    for (const auto& row : rows) {
      if (row.missing || row.data == nullptr || row.size != row_bytes) {
        throw std::runtime_error("RecStore restored an invalid value row");
      }
      int64_t row_index = -1;
      std::memcpy(&row_index, row.data, sizeof(int64_t));
      if (row_index < 0 || row_index >= state.next_row_index ||
          seen[static_cast<size_t>(row_index)] != 0) {
        throw std::runtime_error("RecStore restored invalid stable indices");
      }
      seen[static_cast<size_t>(row_index)] = 1;
    }
    if (std::find(seen.begin(), seen.end(), 0) != seen.end()) {
      throw std::runtime_error("RecStore restored allocator has unowned rows");
    }
  }

  bool WriteIndexedManifest(const std::filesystem::path& checkpoint_path,
                            const std::filesystem::path& temp_path,
                            const std::string& metadata,
                            const std::string& kv_filename,
                            const std::string& state_filename) {
    const uint64_t metadata_size = static_cast<uint64_t>(metadata.size());
    const uint64_t kv_name_size = static_cast<uint64_t>(kv_filename.size());
    const uint64_t state_name_size =
        static_cast<uint64_t>(state_filename.size());
    std::ofstream output(temp_path, std::ios::binary | std::ios::trunc);
    if (!output) {
      return false;
    }
    uint64_t checksum = kIndexedCheckpointChecksumSeed;
    auto write = [&](const void* data, size_t size) {
      UpdateCheckpointChecksum(&checksum, data, size);
      return WriteCheckpointBytes(output, data, size);
    };
    bool ok = write(kIndexedManifestMagic.data(),
                    kIndexedManifestMagic.size()) &&
              write(&kIndexedCheckpointVersion,
                    sizeof(kIndexedCheckpointVersion)) &&
              write(&metadata_size, sizeof(metadata_size)) &&
              write(&kv_name_size, sizeof(kv_name_size)) &&
              write(&state_name_size, sizeof(state_name_size)) &&
              write(metadata.data(), metadata.size()) &&
              write(kv_filename.data(), kv_filename.size()) &&
              write(state_filename.data(), state_filename.size()) &&
              WriteCheckpointPod(output, checksum);
    output.flush();
    ok = ok && output.good();
    output.close();
    if (!ok || output.fail()) {
      return false;
    }
    std::error_code error;
    std::filesystem::rename(temp_path, checkpoint_path, error);
    return !error;
  }

  bool SaveCheckpoint(const std::string& checkpoint,
                      const std::string& metadata) {
    if (checkpoint.empty()) {
      throw std::invalid_argument("RecStore checkpoint path must be non-empty");
    }
    std::lock_guard<std::mutex> lock(mu);
    const std::filesystem::path checkpoint_path(checkpoint);
    const std::string base = checkpoint_path.filename().string();
    if (!IsSafeSidecarFilename(base)) {
      throw std::invalid_argument(
          "RecStore checkpoint path must name a regular manifest file");
    }
    const std::filesystem::path parent = checkpoint_path.parent_path().empty()
                                             ? std::filesystem::path(".")
                                             : checkpoint_path.parent_path();
    std::filesystem::create_directories(parent);
    IndexedCheckpointManifest previous_manifest;
    bool has_previous_manifest = false;
    std::error_code exists_error;
    if (std::filesystem::is_regular_file(checkpoint_path, exists_error) &&
        !exists_error) {
      try {
        previous_manifest = ReadIndexedCheckpointManifest(checkpoint_path);
        has_previous_manifest = true;
      } catch (const std::exception&) {
        // A valid new generation may replace a corrupt old manifest. Its
        // unreferenced sidecars cannot be identified safely and are retained.
      }
    }
    static std::atomic<uint64_t> generation{0};
    const uint64_t nonce = static_cast<uint64_t>(
        std::chrono::steady_clock::now().time_since_epoch().count());
    const std::string suffix = "." + std::to_string(getpid()) + "." +
                               std::to_string(nonce) + "." +
                               std::to_string(generation.fetch_add(1));
    const std::string kv_filename = base + suffix + ".kv";
    const std::string state_filename = base + suffix + ".state";
    const std::filesystem::path kv_path = parent / kv_filename;
    const std::filesystem::path state_path = parent / state_filename;
    const std::filesystem::path manifest_temp =
        parent / (base + suffix + ".manifest.tmp");

    if (!kv->SaveCheckpoint(kv_path.string(), metadata) ||
        !WriteIndexedState(state_path, metadata) ||
        !WriteIndexedManifest(checkpoint_path, manifest_temp, metadata,
                              kv_filename, state_filename)) {
      std::error_code error;
      std::filesystem::remove(kv_path, error);
      std::filesystem::remove(state_path, error);
      std::filesystem::remove(manifest_temp, error);
      return false;
    }
    if (has_previous_manifest) {
      std::error_code cleanup_error;
      if (previous_manifest.kv_filename != kv_filename) {
        std::filesystem::remove(parent / previous_manifest.kv_filename,
                                cleanup_error);
      }
      cleanup_error.clear();
      if (previous_manifest.state_filename != state_filename) {
        std::filesystem::remove(parent / previous_manifest.state_filename,
                                cleanup_error);
      }
    }
    return true;
  }

  bool LoadCheckpoint(const std::string& checkpoint,
                      const std::string& expected_metadata) {
    if (checkpoint.empty()) {
      throw std::invalid_argument("RecStore checkpoint path must be non-empty");
    }
    std::lock_guard<std::mutex> lock(mu);
    if (kv->ActiveKeyCount() != 0 || next_row_index != 0 ||
        !free_row_indices.empty()) {
      throw std::runtime_error(
          "RecStore checkpoint restore requires an empty indexed engine");
    }
    const std::filesystem::path checkpoint_path(checkpoint);
    const IndexedCheckpointManifest manifest =
        ReadIndexedCheckpointManifest(checkpoint_path);
    if (manifest.metadata != expected_metadata) {
      throw std::runtime_error("RecStore checkpoint metadata mismatch");
    }
    const std::filesystem::path parent = checkpoint_path.parent_path().empty()
                                             ? std::filesystem::path(".")
                                             : checkpoint_path.parent_path();
    const std::filesystem::path kv_path = parent / manifest.kv_filename;
    const std::filesystem::path state_path = parent / manifest.state_filename;
    RestoredIndexedState state =
        ReadIndexedState(state_path, expected_metadata);

    std::vector<std::unique_ptr<AdamSlotBlock>> restored_slot_blocks;
    const size_t required_blocks = static_cast<size_t>(
        (state.slot_rows + kAdamSlotBlockRows - 1) / kAdamSlotBlockRows);
    restored_slot_blocks.reserve(required_blocks);
    for (size_t block = 0; block < required_blocks; ++block) {
      restored_slot_blocks.push_back(std::make_unique<AdamSlotBlock>(dim));
    }
    size_t offset = 0;
    int64_t remaining_rows = state.slot_rows;
    for (size_t block = 0; remaining_rows > 0; ++block) {
      const int64_t rows =
          std::min<int64_t>(remaining_rows, kAdamSlotBlockRows);
      const size_t count =
          static_cast<size_t>(rows) * static_cast<size_t>(dim);
      std::memcpy(restored_slot_blocks[block]->exp_avg.data(),
                  state.exp_avg.data() + offset, count * sizeof(float));
      std::memcpy(restored_slot_blocks[block]->exp_avg_sq.data(),
                  state.exp_avg_sq.data() + offset, count * sizeof(float));
      offset += count;
      remaining_rows -= rows;
    }
    if (!kv->LoadCheckpoint(kv_path.string(), expected_metadata)) {
      return false;
    }
    try {
      ValidateRestoredRows(state);
    } catch (...) {
      kv->clear();
      throw;
    }

    next_row_index = state.next_row_index;
    free_row_indices = std::move(state.free_rows);
    adam_step = state.adam_step;
    adam_beta1_t = state.adam_beta1_t;
    adam_beta2_t = state.adam_beta2_t;
    adam_slot_blocks = std::move(restored_slot_blocks);
    return true;
  }

  void PutIndexed(
      const py::array_t<int64_t, py::array::c_style | py::array::forcecast>& ids,
      const py::array_t<int64_t, py::array::c_style | py::array::forcecast>& indices,
      const py::array_t<float, py::array::c_style | py::array::forcecast>& values) {
    const auto keys = CheckIdsView(ids);
    auto index_view = indices.unchecked<1>();
    auto value_view = values.unchecked<2>();
    if (index_view.shape(0) != static_cast<py::ssize_t>(keys.size()) ||
        value_view.shape(0) != static_cast<py::ssize_t>(keys.size()) ||
        value_view.shape(1) != dim) {
      throw std::invalid_argument(
          "put_indexed: ids, indices, and [num_ids, dim] values must align");
    }

    std::lock_guard<std::mutex> lock(mu);
    std::vector<float> packed(keys.size() * static_cast<size_t>(storage_dim));
    std::vector<base::ConstArray<float>> arrays(keys.size());
    for (size_t i = 0; i < keys.size(); ++i) {
      const int64_t row_index = index_view(static_cast<py::ssize_t>(i));
      if (row_index < 0 || row_index >= capacity) {
        throw std::invalid_argument("put_indexed: row index is out of range");
      }
      float* row = packed.data() + i * static_cast<size_t>(storage_dim);
      std::memcpy(row, &row_index, sizeof(int64_t));
      std::memcpy(row + sizeof(int64_t) / sizeof(float),
                  value_view.data(static_cast<py::ssize_t>(i), 0),
                  static_cast<size_t>(dim) * sizeof(float));
      arrays[i] = base::ConstArray<float>(row, storage_dim);
      next_row_index = std::max(next_row_index, row_index + 1);
    }
    if (!keys.empty()) {
      kv->BatchPut(keys.AsConstArray(), &arrays, 0);
    }
  }

  py::dict BulkInsertIndexed(
      const py::array_t<int64_t, py::array::c_style | py::array::forcecast>& ids,
      const py::array_t<int64_t, py::array::c_style | py::array::forcecast>& indices,
      const py::array_t<float, py::array::c_style | py::array::forcecast>& values,
      bool release_gil) {
    using clock = std::chrono::steady_clock;
    const auto total_start = clock::now();
    const auto keys = CheckIdsView(ids);
    auto index_view = indices.unchecked<1>();
    auto value_view = values.unchecked<2>();
    if (index_view.shape(0) != static_cast<py::ssize_t>(keys.size()) ||
        value_view.shape(0) != static_cast<py::ssize_t>(keys.size()) ||
        value_view.shape(1) != dim) {
      throw std::invalid_argument(
          "bulk_insert_indexed: ids, indices, and [num_ids, dim] values "
          "must align");
    }
    const auto convert_end = clock::now();

    int invalid_index = 0;
    int64_t max_row_index = -1;
    const int workers = EffectiveWorkers(keys.size());
#pragma omp parallel for num_threads(workers) schedule(static) \
    reduction(| : invalid_index) reduction(max : max_row_index)
    for (int64_t i = 0; i < static_cast<int64_t>(keys.size()); ++i) {
      const int64_t row_index = index_view(static_cast<py::ssize_t>(i));
      if (row_index < 0 || row_index >= capacity) {
        invalid_index = 1;
        continue;
      }
      max_row_index = std::max(max_row_index, row_index);
    }
    if (invalid_index != 0) {
      throw std::invalid_argument(
          "bulk_insert_indexed: row index is out of range");
    }
    const auto pack_end = clock::now();

    const auto engine_start = clock::now();
    std::vector<BaseKV::BulkLoadStats> worker_stats(
        static_cast<size_t>(workers));
    std::vector<uint8_t> worker_ok(static_cast<size_t>(workers), 1);
    {
      std::lock_guard<std::mutex> lock(mu);
      auto insert = [&]() {
        // This path is intentionally restricted to unique keys that a
        // preceding lookup reserved as missing. Each worker owns a disjoint
        // range, so the engine can allocate values and build the index in
        // parallel without generic overwrite/retire bookkeeping.
#pragma omp parallel for num_threads(workers) schedule(static)
        for (int worker = 0; worker < workers; ++worker) {
          const size_t begin = keys.size() * static_cast<size_t>(worker) /
                               static_cast<size_t>(workers);
          const size_t end = keys.size() * static_cast<size_t>(worker + 1) /
                             static_cast<size_t>(workers);
          if (begin == end) continue;
          worker_ok[static_cast<size_t>(worker)] =
              kv->BulkLoadIndexedFloatRange(
                  base::ConstArray<uint64_t>(
                      keys.data() + begin, static_cast<int>(end - begin)),
                  index_view.data(static_cast<py::ssize_t>(begin)),
                  value_view.data(static_cast<py::ssize_t>(begin), 0),
                  static_cast<int64_t>(end - begin), dim,
                  static_cast<unsigned>(worker),
                  &worker_stats[static_cast<size_t>(worker)])
                  ? 1
                  : 0;
        }
      };
      if (release_gil) {
        py::gil_scoped_release nogil;
        insert();
      } else {
        insert();
      }
      if (!std::all_of(worker_ok.begin(), worker_ok.end(),
                       [](uint8_t ok) { return ok != 0; })) {
        throw std::runtime_error(
            "BulkLoadIndexedFloatRange failed for local indexed rows");
      }
      next_row_index = std::max(next_row_index, max_row_index + 1);
    }
    const auto engine_end = clock::now();
    auto ns = [](clock::duration d) {
      return static_cast<double>(
          std::chrono::duration_cast<std::chrono::nanoseconds>(d).count());
    };
    py::dict profile;
    profile["arg_convert_ns"] = ns(convert_end - total_start);
    profile["pack_ns"] = ns(pack_end - convert_end);
    profile["engine_insert_ns"] = ns(engine_end - engine_start);
    for (const auto& field : {
             std::make_pair("prepare", &BaseKV::BulkLoadStats::prepare_ns),
             std::make_pair(
                 "value_alloc", &BaseKV::BulkLoadStats::value_alloc_ns),
             std::make_pair(
                 "handle_validate",
                 &BaseKV::BulkLoadStats::handle_validate_ns),
             std::make_pair("index_put", &BaseKV::BulkLoadStats::index_put_ns),
             std::make_pair("key_track", &BaseKV::BulkLoadStats::key_track_ns),
             std::make_pair("worker_total", &BaseKV::BulkLoadStats::total_ns),
         }) {
      uint64_t cpu_ns = 0;
      uint64_t max_ns = 0;
      for (const auto& worker_stat : worker_stats) {
        const uint64_t value = worker_stat.*(field.second);
        cpu_ns += value;
        max_ns = std::max(max_ns, value);
      }
      profile[py::str(std::string("bulk_") + field.first + "_cpu_ns")] =
          static_cast<double>(cpu_ns);
      profile[py::str(std::string("bulk_") + field.first + "_max_ns")] =
          static_cast<double>(max_ns);
    }
    profile["total_ns"] = ns(engine_end - total_start);
    profile["rows"] = static_cast<double>(keys.size());
    profile["workers"] = static_cast<double>(workers);
    return profile;
  }

  py::dict SgdUpdateIndexed(
      const py::array_t<int64_t, py::array::c_style | py::array::forcecast>& ids,
      const py::array_t<float, py::array::c_style | py::array::forcecast>& grads,
      double lr,
      bool release_gil) {
    using clock = std::chrono::steady_clock;
    const auto total_start = clock::now();
    const auto keys = CheckIdsView(ids);
    const auto convert_end = clock::now();
    auto g = grads.unchecked<2>();
    if (g.shape(0) != static_cast<py::ssize_t>(keys.size()) ||
        g.shape(1) != dim) {
      throw std::invalid_argument(
          "sgd_update_indexed: grads must be [num_ids, dim] float");
    }
    if (!(lr > 0.0)) {
      throw std::invalid_argument("sgd_update_indexed: lr must be positive");
    }

    const auto engine_start = clock::now();
    bool ok = true;
    {
      std::lock_guard<std::mutex> lock(mu);
      if (release_gil) {
        py::gil_scoped_release nogil;
        if (!keys.empty())
          ok = ApplySgdUpdateParallel(keys, g.data(0, 0),
                                      static_cast<float>(lr));
      } else if (!keys.empty()) {
        ok = ApplySgdUpdateParallel(keys, g.data(0, 0),
                                    static_cast<float>(lr));
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
    profile["arg_convert_ns"] = ns(convert_end - total_start);
    profile["engine_update_ns"] = ns(engine_end - engine_start);
    profile["total_ns"] = ns(engine_end - total_start);
    profile["rows"] = static_cast<double>(keys.size());
    profile["workers"] = static_cast<double>(EffectiveWorkers(keys.size()));
    return profile;
  }

  py::dict AdamWTFUpdateIndexed(
      const py::array_t<int64_t, py::array::c_style | py::array::forcecast>& ids,
      const py::array_t<float, py::array::c_style | py::array::forcecast>& grads,
      double lr,
      double beta1,
      double beta2,
      double eps,
      double weight_decay,
      bool release_gil) {
    using clock = std::chrono::steady_clock;
    const auto total_start = clock::now();
    const auto keys = CheckIdsView(ids);
    const auto convert_end = clock::now();
    auto g = grads.unchecked<2>();
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
    bool ok = true;
    AdamUpdateStats update_stats;
    {
      std::lock_guard<std::mutex> lock(mu);
      const auto lock_end = clock::now();
      if (release_gil) {
        py::gil_scoped_release nogil;
        if (!keys.empty()) {
          ok = ApplyAdamWTFUpdateParallel(
              keys, g.data(0, 0), lr, beta1, beta2, eps, weight_decay,
              &update_stats);
        }
      } else if (!keys.empty()) {
        ok = ApplyAdamWTFUpdateParallel(
            keys, g.data(0, 0), lr, beta1, beta2, eps, weight_decay,
            &update_stats);
      }
      if (!ok) {
        throw std::runtime_error(
            "AdamWTF update failed (keys must exist and stable indices must "
            "be valid)");
      }
      const auto engine_end = clock::now();
      auto ns = [](clock::duration d) {
        return static_cast<double>(
            std::chrono::duration_cast<std::chrono::nanoseconds>(d).count());
      };
      py::dict profile;
      profile["arg_convert_ns"] = ns(convert_end - total_start);
      profile["lock_wait_ns"] = ns(lock_end - lock_start);
      profile["engine_update_ns"] = ns(engine_end - lock_end);
      profile["engine_get_ns"] =
          static_cast<double>(update_stats.engine_get_ns);
      profile["engine_index_lookup_ns"] =
          static_cast<double>(update_stats.index_lookup_ns);
      profile["engine_index_lookup_max_ns"] =
          static_cast<double>(update_stats.index_lookup_max_ns);
      profile["slot_alloc_ns"] =
          static_cast<double>(update_stats.slot_alloc_ns);
      profile["adam_math_ns"] = static_cast<double>(update_stats.math_ns);
      profile["total_ns"] = ns(engine_end - total_start);
      profile["rows"] = static_cast<double>(keys.size());
      profile["workers"] =
          static_cast<double>(EffectiveWorkers(keys.size()));
      profile["step"] = static_cast<double>(adam_step);
      return profile;
    }
  }

  py::dict OptimizerStateStats() const {
    py::dict stats;
    stats["adam_step"] = static_cast<double>(adam_step);
    stats["adam_beta1_t"] = static_cast<double>(adam_beta1_t);
    stats["adam_beta2_t"] = static_cast<double>(adam_beta2_t);
    stats["adam_slot_blocks"] =
        static_cast<double>(adam_slot_blocks.size());
    stats["adam_slot_rows"] = static_cast<double>(
        adam_slot_blocks.size() * static_cast<size_t>(kAdamSlotBlockRows));
    stats["adam_slot_bytes"] = static_cast<double>(
        adam_slot_blocks.size() * static_cast<size_t>(kAdamSlotBlockRows) *
        static_cast<size_t>(dim) * sizeof(float) * 2);
    return stats;
  }

  py::dict StorageStats() {
    std::lock_guard<std::mutex> lock(mu);
    const uint64_t active_rows = kv->ActiveKeyCount();
    const uint64_t stable_rows = static_cast<uint64_t>(next_row_index);
    const uint64_t free_rows = static_cast<uint64_t>(free_row_indices.size());
    const uint64_t embedding_row_bytes =
        static_cast<uint64_t>(dim) * sizeof(float);
    const uint64_t value_row_bytes =
        static_cast<uint64_t>(storage_dim) * sizeof(float);
    const uint64_t adam_bytes =
        static_cast<uint64_t>(adam_slot_blocks.size()) *
        static_cast<uint64_t>(kAdamSlotBlockRows) *
        static_cast<uint64_t>(dim) * sizeof(float) * 2;
    py::dict stats;
    stats["capacity"] = static_cast<double>(capacity);
    stats["active_rows"] = static_cast<double>(active_rows);
    stats["stable_rows"] = static_cast<double>(stable_rows);
    stats["free_rows"] = static_cast<double>(free_rows);
    stats["embedding_row_bytes"] = static_cast<double>(embedding_row_bytes);
    stats["value_row_bytes"] = static_cast<double>(value_row_bytes);
    stats["adam_slot_bytes"] = static_cast<double>(adam_bytes);
    stats["estimated_embedding_bytes"] =
        static_cast<double>(stable_rows * embedding_row_bytes);
    stats["estimated_value_bytes"] =
        static_cast<double>(active_rows * value_row_bytes);
    return stats;
  }
};

// RecStore's base objects register two gflags (int/double) via
// google::FlagRegisterer. This .so must be LD_PRELOADable together with
// processes that embed their own gflags (e.g. a bundled brpc), so instead of
// depending on a shared libgflags it satisfies those two registration sites
// with hidden no-op constructors (defaults live in the flag globals). The
// gflags headers RecStore compiles against already declare the class, so
// redefine via raw mangled names rather than redeclaring the type.
extern "C" {
__attribute__((visibility("hidden"))) void _ZN6google14FlagRegistererC1IiEEPKcS3_S3_PT_S5_(
    const char*, const char*, const char*, int*, int*) {}
__attribute__((visibility("hidden"))) void _ZN6google14FlagRegistererC1IdEEPKcS3_S3_PT_S5_(
    const char*, const char*, const char*, double*, double*) {}
}

PYBIND11_MODULE(_local_kv, m) {
  py::class_<LocalKV>(m, "LocalKV")
      .def(py::init<int64_t, int64_t, const std::string&,
                    const std::string&, const std::string&, int64_t, int>(),
           py::arg("capacity"), py::arg("embedding_dim"),
           py::arg("index_type"), py::arg("value_path"),
           py::arg("allocator") = "CONCURRENT_SLAB_MEMORY_POOL",
           py::arg("allocator_capacity") = 0,
           py::arg("num_threads") = 1)
      .def("lookup", &LocalKV::Lookup)
      .def("lookup_profiled", &LocalKV::LookupProfiled)
      .def("index_only", &LocalKV::IndexOnly)
      .def("lookup_into", &LocalKV::LookupInto)
      .def("put", &LocalKV::Put)
      .def("sgd_update", &LocalKV::SgdUpdate,
           py::arg("ids"), py::arg("grads"), py::arg("lr"),
           py::arg("release_gil") = true)
      .def("adamw_update", &LocalKV::AdamWUpdate,
           py::arg("ids"), py::arg("grads"), py::arg("lr"),
           py::arg("beta1"), py::arg("beta2"), py::arg("eps"),
           py::arg("weight_decay"), py::arg("release_gil") = true);

  py::class_<LocalIndexedKV>(m, "LocalIndexedKV")
      .def(py::init<int64_t, int64_t, const std::string&,
                    const std::string&, const std::string&, int64_t, int, int>(),
           py::arg("capacity"), py::arg("embedding_dim"),
           py::arg("index_type"), py::arg("value_path"),
           py::arg("allocator") = "CONCURRENT_SLAB_MEMORY_POOL",
           py::arg("allocator_capacity") = 0,
           py::arg("num_threads") = 1,
           py::arg("index_prefetch") = 1)
      .def("lookup_indexed", &LocalIndexedKV::LookupIndexed,
           py::arg("ids"), py::arg("allocate_missing") = true)
      .def("lookup_indexed_profiled", &LocalIndexedKV::LookupIndexedProfiled,
           py::arg("ids"), py::arg("allocate_missing") = true)
      .def("lookup_indexed_into_raw", &LocalIndexedKV::LookupIndexedIntoRaw,
           py::arg("ids_address"), py::arg("num_rows"),
           py::arg("indices_address"), py::arg("values_address"),
           py::arg("found_address"), py::arg("allocate_missing") = true,
           py::arg("release_gil") = true,
           py::arg("collect_profile") = false)
      .def("lookup_indexed_fused_into_raw", &LocalIndexedKV::LookupIndexedFusedIntoRaw,
           py::arg("ids_address"), py::arg("num_rows"),
           py::arg("indices_address"), py::arg("values_address"),
           py::arg("found_address"), py::arg("allocate_missing") = true,
           py::arg("release_gil") = true,
           py::arg("collect_profile") = false)
      .def("lookup_indexed_legacy_into_raw", &LocalIndexedKV::LookupIndexedLegacyIntoRaw,
           py::arg("ids_address"), py::arg("num_rows"),
           py::arg("indices_address"), py::arg("values_address"),
           py::arg("found_address"), py::arg("allocate_missing") = true,
           py::arg("release_gil") = true,
           py::arg("collect_profile") = false)
      .def("lookup_values_into_raw", &LocalIndexedKV::LookupValuesIntoRaw,
           py::arg("ids_address"), py::arg("num_rows"),
           py::arg("values_address"), py::arg("release_gil") = true)
      .def("lookup_values_into_raw_profiled",
           &LocalIndexedKV::LookupValuesIntoRawProfiled,
           py::arg("ids_address"), py::arg("num_rows"),
           py::arg("values_address"), py::arg("release_gil") = true)
      .def("lookup_values_with_misses_into_raw",
           &LocalIndexedKV::LookupValuesWithMissesIntoRaw,
           py::arg("ids_address"), py::arg("num_rows"),
           py::arg("values_address"), py::arg("missing_positions_address"),
           py::arg("release_gil") = true)
      .def("lookup_values_with_misses_into_raw_profiled",
           &LocalIndexedKV::LookupValuesWithMissesIntoRawProfiled,
           py::arg("ids_address"), py::arg("num_rows"),
           py::arg("values_address"), py::arg("missing_positions_address"),
           py::arg("release_gil") = true)
      .def("size", &LocalIndexedKV::Size)
      .def("snapshot_indexed", &LocalIndexedKV::SnapshotIndexed)
      .def("delete_indexed", &LocalIndexedKV::DeleteIndexed)
      .def("clear_indexed", &LocalIndexedKV::ClearIndexed)
      .def("save_checkpoint", &LocalIndexedKV::SaveCheckpoint,
           py::arg("path"), py::arg("metadata"))
      .def("load_checkpoint", &LocalIndexedKV::LoadCheckpoint,
           py::arg("path"), py::arg("expected_metadata"))
      .def("put_indexed", &LocalIndexedKV::PutIndexed)
      .def("bulk_insert_indexed", &LocalIndexedKV::BulkInsertIndexed,
           py::arg("ids"), py::arg("indices"), py::arg("values"),
           py::arg("release_gil") = true)
      .def("sgd_update_indexed", &LocalIndexedKV::SgdUpdateIndexed,
           py::arg("ids"), py::arg("grads"), py::arg("lr"),
           py::arg("release_gil") = true)
      .def("adamw_tf_update_indexed", &LocalIndexedKV::AdamWTFUpdateIndexed,
           py::arg("ids"), py::arg("grads"), py::arg("lr"),
           py::arg("beta1"), py::arg("beta2"), py::arg("eps"),
           py::arg("weight_decay"), py::arg("release_gil") = true)
      .def("optimizer_state_stats", &LocalIndexedKV::OptimizerStateStats)
      .def("storage_stats", &LocalIndexedKV::StorageStats);
}
