#pragma once

#include <algorithm>
#include <cmath>
#include "base/array.h"
#include "base/tensor.h"
#include <cstdint>
#include <cstring>
#include <functional>
#include <limits>
#include <mutex>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <vector>

namespace recstore {

// Homogeneous-host wire format, shared by RPC, SHM and RDMA transports.
struct TrainingStepHeader {
  uint64_t session = 0;
  int64_t step = 0;
  uint32_t rank = 0;
  uint32_t world = 1;
  uint32_t dim = 0;
  uint32_t rows = 0;
  uint32_t chunk = 0;
  uint32_t chunks = 1;
  uint32_t kind = 0;  // 0: submit, 1: query, 2: close, 3: abort
  float learning_rate = 0.01f;
  uint32_t divisor = 1;
  uint32_t version = 1;
};
static_assert(sizeof(TrainingStepHeader) == 56, "TrainingStepHeader wire size");

inline TrainingStepHeader DecodeTrainingStep(const std::string& payload) {
  TrainingStepHeader h;
  if (payload.size() < sizeof(h)) {
    throw std::invalid_argument("Truncated training step header");
  }
  std::memcpy(&h, payload.data(), sizeof(h));
  if (h.version != 1 || !h.session || !h.world || h.world > 65536 ||
      h.rank >= h.world || !h.dim || h.dim > 65536 || h.step < 0 ||
      h.kind > 3 || !h.chunks || h.chunks > 1048576 || h.chunk >= h.chunks ||
      (h.divisor != 1 && h.divisor != h.world) ||
      !std::isfinite(h.learning_rate) || h.learning_rate < 0) {
    throw std::invalid_argument("Invalid training step metadata");
  }
  const uint64_t bytes = sizeof(h) + uint64_t(h.rows) * (8 + uint64_t(h.dim) * 4);
  if (bytes != payload.size() || (h.kind != 0 && h.rows != 0)) {
    throw std::invalid_argument("Invalid training step payload size");
  }
  return h;
}

inline std::string PackTrainingStep(TrainingStepHeader h,
                                    const uint64_t* keys = nullptr,
                                    const float* grads = nullptr) {
  std::string result(sizeof(h) + size_t(h.rows) * (8 + size_t(h.dim) * 4), '\0');
  std::memcpy(result.data(), &h, sizeof(h));
  if (h.rows) {
    if (!keys || !grads) throw std::invalid_argument("Missing training step rows");
    std::memcpy(result.data() + sizeof(h), keys, size_t(h.rows) * 8);
    std::memcpy(result.data() + sizeof(h) + size_t(h.rows) * 8,
                grads, size_t(h.rows) * h.dim * 4);
  }
  DecodeTrainingStep(result);
  return result;
}

class TrainingStepClient {
 public:
  virtual ~TrainingStepClient() = default;
  virtual int64_t ExchangeTrainingStep(const std::string& table,
                                       const std::string& payload) {
    throw std::runtime_error("Training step protocol unsupported by this PS backend");
  }

  virtual size_t TrainingStepPayloadLimit() const { return 256 * 1024; }

  virtual int64_t TrainingStep(const std::string& table, TrainingStepHeader h,
                               const base::ConstArray<uint64_t>& keys,
                               const base::RecTensor& grads) {
    if (h.kind != 0) return ExchangeTrainingStep(table, PackTrainingStep(h));
    if (grads.dim() != 2 || grads.shape(0) != static_cast<int64_t>(keys.Size()) ||
        grads.shape(1) != h.dim || grads.dtype() != base::DataType::FLOAT32) {
      throw std::invalid_argument("Invalid training gradient shape");
    }
    const size_t limit = TrainingStepPayloadLimit();
    const size_t row_bytes = 8 + size_t(h.dim) * 4;
    if (limit < sizeof(h) + row_bytes) throw std::runtime_error("Training row exceeds transport slot");
    const size_t rows = (limit - sizeof(h)) / row_bytes;
    h.chunks = std::max<size_t>(1, (keys.Size() + rows - 1) / rows);
    int64_t committed = -1;
    for (h.chunk = 0; h.chunk < h.chunks; ++h.chunk) {
      const size_t begin = size_t(h.chunk) * rows;
      h.rows = std::min(rows, keys.Size() - begin);
      committed = ExchangeTrainingStep(table, PackTrainingStep(
          h, h.rows ? keys.Data() + begin : nullptr,
          h.rows ? grads.data_as<float>() + begin * h.dim : nullptr));
    }
    return committed;
  }

};

// One in-flight step per table. RPC handlers acknowledge chunks immediately;
// they never block a server worker waiting for another trainer.
class TrainingStepCoordinator {
 public:
  using Apply = std::function<void(const std::vector<uint64_t>&,
                                   const std::vector<float>&)>;

  int64_t Exchange(const std::string& table, const std::string& payload,
                   const Apply& apply) {
    const auto h = DecodeTrainingStep(payload);
    // ponytail: one lock per PS shard; split by table if independent jobs need
    // concurrent commits. The fused BagPipe table already requires ordering.
    std::lock_guard<std::mutex> guard(mu_);
    auto it = states_.find(table);
    if (it == states_.end()) {
      if (h.kind != 0 || h.step != 0) {
        throw std::runtime_error("Unknown training session");
      }
      State state;
      state.header = h;
      state.ranks.resize(h.world);
      state.closed.resize(h.world, false);
      it = states_.emplace(table, std::move(state)).first;
    }
    auto& state = it->second;
    const auto& initial = state.header;
    if (initial.session != h.session || initial.world != h.world ||
        initial.dim != h.dim || initial.learning_rate != h.learning_rate ||
        initial.divisor != h.divisor) {
      throw std::runtime_error("Training session configuration mismatch");
    }
    if (h.kind == 3) state.failed = true;
    if (state.failed) throw std::runtime_error("Training session aborted or commit failed");
    if (h.kind == 1) return state.committed;
    if (h.kind == 2) {
      if (state.committed < h.step) throw std::runtime_error("Closing an uncommitted step");
      state.closed[h.rank] = true;
      const auto committed = state.committed;
      bool all_closed = true;
      for (bool closed : state.closed) all_closed &= closed;
      if (all_closed) states_.erase(it);
      return committed;
    }
    // A transport retry must never apply an already committed step twice.
    if (h.step <= state.committed) return state.committed;
    if (h.step != state.committed + 1 || state.closed[h.rank]) {
      throw std::runtime_error("Training steps must be submitted in order");
    }
    auto& chunks = state.ranks[h.rank];
    if (chunks.empty()) chunks.resize(h.chunks);
    if (chunks.size() != h.chunks) throw std::runtime_error("Training chunk count changed");
    if (!chunks[h.chunk].empty()) {
      if (chunks[h.chunk] != payload) throw std::runtime_error("Conflicting training chunk retry");
      return state.committed;
    }
    chunks[h.chunk] = payload;
    for (const auto& rank : state.ranks) {
      if (rank.empty()) return state.committed;
      for (const auto& chunk : rank) if (chunk.empty()) return state.committed;
    }
    std::vector<uint64_t> keys;
    std::vector<float> grads;
    std::unordered_map<uint64_t, size_t> positions;
    // Fixed rank/chunk order makes network arrival order irrelevant.
    for (const auto& rank : state.ranks) {
      for (const auto& chunk : rank) {
        const auto ch = DecodeTrainingStep(chunk);
        const char* key_data = chunk.data() + sizeof(ch);
        const char* grad_data = key_data + size_t(ch.rows) * 8;
        for (size_t row = 0; row < ch.rows; ++row) {
          uint64_t key;
          std::memcpy(&key, key_data + row * 8, 8);
          auto entry = positions.emplace(key, keys.size());
          if (entry.second) {
            keys.push_back(key);
            grads.resize(keys.size() * h.dim, 0.0f);
          }
          float* dst = grads.data() + entry.first->second * h.dim;
          for (size_t col = 0; col < h.dim; ++col) {
            float value;
            std::memcpy(&value, grad_data + (row * h.dim + col) * 4, 4);
            dst[col] += value;
          }
        }
      }
    }
    if (h.divisor != 1) for (auto& value : grads) value /= h.divisor;
    try {
      if (!keys.empty()) apply(keys, grads);
    } catch (...) {
      state.failed = true;
      throw;
    }
    state.committed = h.step;
    for (auto& rank : state.ranks) rank.clear();
    return state.committed;
  }

 private:
  struct State {
    TrainingStepHeader header;
    int64_t committed = -1;
    bool failed = false;
    std::vector<std::vector<std::string>> ranks;
    std::vector<bool> closed;
  };
  std::mutex mu_;
  std::unordered_map<std::string, State> states_;
};
} // namespace recstore
