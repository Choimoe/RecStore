#pragma once

#include <algorithm>
#include <atomic>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <cstdlib>
#include <mutex>
#include <string>
#include <string_view>
#include <vector>

#include "base/log.h"
#include "third_party/Mayfly-main/include/GlobalAddress.h"

namespace petps {

inline constexpr std::uint32_t kPutPayloadMagic       = 0x50545053;
inline constexpr std::uint16_t kPutProtocolVersionV1  = 1;
inline constexpr std::uint16_t kPutProtocolVersionV2  = 2;
inline constexpr std::uint32_t kPutRemotePayloadMagic = 0x50545232;
inline constexpr std::uint32_t kPutV2TransferModeRead = 0;
inline constexpr std::uint32_t kPutV2TransferModePush = 1;

struct PutPayloadHeader {
  std::uint32_t magic;
  std::uint16_t version;
  std::uint16_t reserved;
  std::uint32_t key_count;
  std::uint32_t embedding_dim;
};

struct DecodedPutPayload {
  std::uint32_t embedding_dim = 0;
  std::vector<std::uint64_t> keys;
  std::vector<float> values;
};

struct PutRemotePayloadV2 {
  std::uint32_t magic;
  std::uint16_t version;
  std::uint16_t reserved;
  std::uint32_t key_count;
  std::uint32_t embedding_dim;
  GlobalAddress payload_gaddr;
  std::uint32_t payload_bytes;
  std::uint32_t transfer_mode;
  std::uint32_t checksum;
} __attribute__((packed));

static_assert(sizeof(PutRemotePayloadV2) <= 64, "unexpected v2 control size");

inline constexpr std::uint32_t kRdmaDescriptorMagic     = 0x52444431;
inline constexpr std::uint16_t kRdmaDescriptorVersionV1 = 1;

enum class RdmaDescriptorOp : std::uint16_t {
  kGet = 1,
  kPut = 2,
};

struct RdmaDescriptorRequest {
  std::uint32_t magic            = kRdmaDescriptorMagic;
  std::uint16_t version          = kRdmaDescriptorVersionV1;
  std::uint16_t op               = 0;
  std::uint64_t request_id       = 0;
  std::uint16_t client_node_id   = 0;
  std::uint16_t client_thread_id = 0;
  std::uint32_t lane_id          = 0;
  std::uint32_t slot_id          = 0;
  std::uint32_t key_count        = 0;
  std::uint32_t embedding_dim    = 0;
  GlobalAddress descriptor_gaddr;
  GlobalAddress response_gaddr;
  GlobalAddress status_gaddr;
  std::uint32_t payload_bytes  = 0;
  std::uint32_t response_bytes = 0;
} __attribute__((packed));

struct RdmaDescriptorLaneConfig {
  std::uint64_t region_offset    = 0;
  std::uint64_t slot_bytes       = 0;
  std::uint32_t slots_per_client = 0;
  std::uint32_t machine_count    = 0;
};

struct RdmaDescriptorDoorbellPostPlan {
  bool use_write_with_imm               = true;
  bool signal_write_with_imm            = false;
  bool wait_write_with_imm_completion   = false;
  bool signal_descriptor_write          = false;
  bool wait_descriptor_write_completion = false;
  bool signal_doorbell                  = false;
  bool wait_doorbell_completion         = false;
};

struct RdmaDescriptorPushPayloadPostPlan {
  bool use_raw_write                 = true;
  bool signal_payload_write          = false;
  bool wait_payload_write_completion = false;
};

inline constexpr std::uint64_t kRdmaDescriptorSignalBatchSize = 32;
inline constexpr std::uint64_t kRdmaDescriptorSignalBatchMask =
    kRdmaDescriptorSignalBatchSize - 1;

struct RdmaDescriptorDoorbellPostDecision {
  bool poll_before_post      = false;
  bool signal_write_with_imm = false;
};

struct RdmaDescriptorDsmWriteDecision {
  bool poll_before_write = false;
  bool signal_write      = false;
};

enum class RdmaDescriptorClientCompletionMode {
  kWaitForCompletion,
  kReturnAfterPost,
};

static_assert(sizeof(RdmaDescriptorRequest) <= 128,
              "descriptor request must fit a small doorbell slot");

inline constexpr RdmaDescriptorDoorbellPostPlan
GetRdmaDescriptorDoorbellPostPlan() {
  return RdmaDescriptorDoorbellPostPlan{};
}

inline constexpr RdmaDescriptorPushPayloadPostPlan
GetRdmaDescriptorPushPayloadPostPlan() {
  return RdmaDescriptorPushPayloadPostPlan{};
}

inline constexpr RdmaDescriptorClientCompletionMode
GetRdmaDescriptorGetCompletionMode(bool is_async) {
  return is_async ? RdmaDescriptorClientCompletionMode::kReturnAfterPost
                  : RdmaDescriptorClientCompletionMode::kWaitForCompletion;
}

inline constexpr RdmaDescriptorDoorbellPostDecision
GetRdmaDescriptorDoorbellPostDecision(std::uint64_t send_counter) {
  const bool signal = (send_counter & kRdmaDescriptorSignalBatchMask) == 0;
  return RdmaDescriptorDoorbellPostDecision{
      signal && send_counter > 0,
      signal,
  };
}

class RdmaDescriptorDoorbellPostState {
public:
  RdmaDescriptorDoorbellPostDecision Next(int lane_id) {
    CHECK_GE(lane_id, 0);
    std::lock_guard<std::mutex> guard(mu_);
    const auto index = static_cast<std::size_t>(lane_id);
    if (index >= send_counters_by_lane_.size()) {
      send_counters_by_lane_.resize(index + 1, 0);
    }
    return GetRdmaDescriptorDoorbellPostDecision(
        send_counters_by_lane_[index]++);
  }

private:
  std::mutex mu_;
  std::vector<std::uint64_t> send_counters_by_lane_;
};

inline constexpr RdmaDescriptorDsmWriteDecision
GetRdmaDescriptorDsmWriteDecision(std::uint64_t send_counter) {
  const bool signal = (send_counter & kRdmaDescriptorSignalBatchMask) == 0;
  return RdmaDescriptorDsmWriteDecision{
      signal && send_counter > 0,
      signal,
  };
}

inline constexpr bool CanPublishRdmaDescriptorReady(
    int registered_threads, int thread_count, bool raw_connected) {
  return thread_count > 0 && registered_threads >= thread_count &&
         raw_connected;
}

inline int
SelectRdmaDescriptorWorkerThread(int thread_count, std::uint64_t cursor) {
  if (thread_count <= 1) {
    return 0;
  }
  return 1 + static_cast<int>(
                 cursor % static_cast<std::uint64_t>(thread_count - 1));
}

inline std::vector<int> GetRdmaDescriptorServingThreadIDs(int thread_count) {
  std::vector<int> thread_ids;
  if (thread_count <= 1) {
    thread_ids.push_back(0);
    return thread_ids;
  }
  thread_ids.reserve(static_cast<std::size_t>(thread_count - 1));
  for (int i = 1; i < thread_count; ++i) {
    thread_ids.push_back(i);
  }
  return thread_ids;
}

inline std::vector<int> RotateRdmaDescriptorServingThreadIDs(
    const std::vector<int>& thread_ids, std::uint64_t seed) {
  if (thread_ids.empty()) {
    return {};
  }
  std::vector<int> rotated = thread_ids;
  const auto shift         = static_cast<std::ptrdiff_t>(seed % rotated.size());
  std::rotate(rotated.begin(), rotated.begin() + shift, rotated.end());
  return rotated;
}

inline std::string RdmaDescriptorWorkerThreadsKey(int server_id) {
  return "petps-descriptor-workers-" + std::to_string(server_id);
}

inline bool TrySelectRdmaDescriptorWorkerThread(
    int thread_count,
    std::uint32_t lane_id,
    int* worker_thread,
    std::string* error) {
  if (worker_thread == nullptr) {
    if (error != nullptr) {
      *error = "worker_thread is null";
    }
    return false;
  }
  if (thread_count <= 0) {
    if (error != nullptr) {
      *error = "invalid descriptor thread count";
    }
    return false;
  }
  if (thread_count == 1) {
    if (lane_id != 0) {
      if (error != nullptr) {
        *error = "descriptor lane out of range";
      }
      return false;
    }
    *worker_thread = 0;
    return true;
  }
  if (lane_id == 0 || lane_id >= static_cast<std::uint32_t>(thread_count)) {
    if (error != nullptr) {
      *error = "descriptor lane out of range";
    }
    return false;
  }
  *worker_thread = static_cast<int>(lane_id);
  return true;
}

inline std::string
EncodeRdmaDescriptorWorkerThreads(const std::vector<int>& thread_ids) {
  std::string payload;
  for (std::size_t i = 0; i < thread_ids.size(); ++i) {
    if (i != 0) {
      payload.push_back(',');
    }
    payload += std::to_string(thread_ids[i]);
  }
  return payload;
}

inline bool DecodeRdmaDescriptorWorkerThreads(
    std::string_view payload,
    std::vector<int>* thread_ids,
    std::string* error) {
  if (thread_ids == nullptr) {
    if (error != nullptr) {
      *error = "thread_ids is null";
    }
    return false;
  }

  bool ascii_payload = true;
  for (char ch : payload) {
    if ((ch < '0' || ch > '9') && ch != ',' && ch != '-' && ch != '+') {
      ascii_payload = false;
      break;
    }
  }
  if (ascii_payload) {
    thread_ids->clear();
    std::size_t pos = 0;
    while (pos < payload.size()) {
      const std::size_t comma = payload.find(',', pos);
      const std::size_t end =
          comma == std::string_view::npos ? payload.size() : comma;
      if (end == pos) {
        if (error != nullptr) {
          *error = "empty descriptor worker thread token";
        }
        return false;
      }
      const std::string token(payload.substr(pos, end - pos));
      char* parse_end  = nullptr;
      const long value = std::strtol(token.c_str(), &parse_end, 10);
      if (parse_end == token.c_str() || *parse_end != '\0') {
        if (error != nullptr) {
          *error = "invalid descriptor worker thread token";
        }
        return false;
      }
      thread_ids->push_back(static_cast<int>(value));
      if (comma == std::string_view::npos) {
        break;
      }
      pos = comma + 1;
    }
    return true;
  }

  if (payload.size() % sizeof(int) != 0) {
    if (error != nullptr) {
      *error = "descriptor worker thread payload size mismatch";
    }
    return false;
  }
  thread_ids->resize(payload.size() / sizeof(int));
  if (!thread_ids->empty()) {
    std::memcpy(thread_ids->data(), payload.data(), payload.size());
  }
  return true;
}

inline std::size_t
FixedSlotResponseBytes(std::size_t key_count, std::size_t value_size_bytes) {
  return key_count * value_size_bytes + sizeof(std::int32_t);
}

inline std::size_t DescriptorClientPermanentSlotBytes(
    std::size_t slot_count, std::size_t slot_bytes) {
  return slot_count * (slot_bytes + sizeof(std::atomic<std::int32_t>));
}

inline std::size_t
PutPayloadBytes(std::size_t key_count, std::size_t value_size_bytes) {
  return key_count * (sizeof(std::uint64_t) + value_size_bytes);
}

inline bool IsValidDescriptorOp(std::uint16_t op) {
  return op == static_cast<std::uint16_t>(RdmaDescriptorOp::kGet) ||
         op == static_cast<std::uint16_t>(RdmaDescriptorOp::kPut);
}

inline bool ValidateRdmaDescriptorRequest(const RdmaDescriptorRequest& request,
                                          std::string* error) {
  if (request.magic != kRdmaDescriptorMagic) {
    if (error != nullptr) {
      *error = "bad descriptor magic";
    }
    return false;
  }
  if (request.version != kRdmaDescriptorVersionV1) {
    if (error != nullptr) {
      *error = "bad descriptor version";
    }
    return false;
  }
  if (!IsValidDescriptorOp(request.op)) {
    if (error != nullptr) {
      *error = "bad descriptor op";
    }
    return false;
  }
  if (request.key_count == 0 || request.embedding_dim == 0) {
    if (error != nullptr) {
      *error = "empty descriptor shape";
    }
    return false;
  }
  return true;
}

inline bool EncodeRdmaDescriptorRequest(const RdmaDescriptorRequest& request,
                                        std::string* payload,
                                        std::string* error) {
  if (payload == nullptr) {
    if (error != nullptr) {
      *error = "payload is null";
    }
    return false;
  }
  if (!ValidateRdmaDescriptorRequest(request, error)) {
    return false;
  }
  payload->resize(sizeof(RdmaDescriptorRequest));
  std::memcpy(payload->data(), &request, sizeof(request));
  return true;
}

inline bool WriteRdmaDescriptorRequest(
    const RdmaDescriptorRequest& request,
    void* buffer,
    std::size_t buffer_bytes,
    std::string* error) {
  if (buffer == nullptr) {
    if (error != nullptr) {
      *error = "descriptor buffer is null";
    }
    return false;
  }
  if (buffer_bytes < sizeof(RdmaDescriptorRequest)) {
    if (error != nullptr) {
      *error = "descriptor buffer too small";
    }
    return false;
  }
  if (!ValidateRdmaDescriptorRequest(request, error)) {
    return false;
  }
  std::memcpy(buffer, &request, sizeof(RdmaDescriptorRequest));
  return true;
}

inline bool DecodeRdmaDescriptorRequest(std::string_view payload,
                                        RdmaDescriptorRequest* request,
                                        std::string* error) {
  if (request == nullptr) {
    if (error != nullptr) {
      *error = "descriptor is null";
    }
    return false;
  }
  if (payload.size() != sizeof(RdmaDescriptorRequest)) {
    if (error != nullptr) {
      *error = "descriptor size mismatch";
    }
    return false;
  }
  RdmaDescriptorRequest decoded{};
  std::memcpy(&decoded, payload.data(), sizeof(decoded));
  if (!ValidateRdmaDescriptorRequest(decoded, error)) {
    return false;
  }
  *request = decoded;
  return true;
}

inline bool ValidateDescriptorLane(const RdmaDescriptorRequest& request,
                                   const RdmaDescriptorLaneConfig& config,
                                   std::string* error) {
  if (config.slot_bytes == 0 || config.slots_per_client == 0) {
    if (error != nullptr) {
      *error = "invalid descriptor lane config";
    }
    return false;
  }
  if (request.client_node_id >= config.machine_count) {
    if (error != nullptr) {
      *error = "client node outside descriptor region";
    }
    return false;
  }
  if (request.slot_id >= config.slots_per_client) {
    if (error != nullptr) {
      *error = "slot outside descriptor lane";
    }
    return false;
  }
  const std::uint64_t expected_offset =
      config.region_offset +
      (static_cast<std::uint64_t>(request.client_node_id) *
           config.slots_per_client +
       request.slot_id) *
          config.slot_bytes;
  if (request.descriptor_gaddr.offset != expected_offset) {
    if (error != nullptr) {
      *error = "descriptor not in sender lane";
    }
    return false;
  }
  return true;
}

inline std::string
EncodePutPayload(const std::vector<std::uint64_t>& keys,
                 const std::vector<std::vector<float>>& values) {
  if (keys.empty()) {
    return {};
  }

  // Validate keys and values have the same count
  if (keys.size() != values.size()) {
    LOG(ERROR) << "EncodePutPayload: keys.size()=" << keys.size()
               << " != values.size()=" << values.size();
    return {};
  }

  const std::size_t embedding_dim = values.front().size();
  // Validate all values have the same dimension
  for (std::size_t i = 0; i < values.size(); ++i) {
    if (values[i].size() != embedding_dim) {
      LOG(ERROR) << "EncodePutPayload: values[" << i
                 << "].size()=" << values[i].size()
                 << " != expected embedding_dim=" << embedding_dim;
      return {};
    }
  }

  PutPayloadHeader header{
      kPutPayloadMagic,
      kPutProtocolVersionV1,
      0,
      static_cast<std::uint32_t>(keys.size()),
      static_cast<std::uint32_t>(embedding_dim),
  };

  std::string payload;
  payload.resize(
      sizeof(PutPayloadHeader) + keys.size() * sizeof(std::uint64_t) +
      keys.size() * embedding_dim * sizeof(float));

  char* cursor = payload.data();
  std::memcpy(cursor, &header, sizeof(header));
  cursor += sizeof(header);

  std::memcpy(cursor, keys.data(), keys.size() * sizeof(std::uint64_t));
  cursor += keys.size() * sizeof(std::uint64_t);

  for (const auto& row : values) {
    std::memcpy(cursor, row.data(), embedding_dim * sizeof(float));
    cursor += embedding_dim * sizeof(float);
  }

  return payload;
}

inline bool DecodePutPayload(
    std::string_view payload, DecodedPutPayload* decoded, std::string* error) {
  if (payload.size() < sizeof(PutPayloadHeader)) {
    if (error != nullptr) {
      *error = "payload smaller than header";
    }
    return false;
  }

  PutPayloadHeader header{};
  std::memcpy(&header, payload.data(), sizeof(header));

  if (header.magic != kPutPayloadMagic) {
    if (error != nullptr) {
      *error = "bad payload magic";
    }
    return false;
  }
  if (header.version != kPutProtocolVersionV1) {
    if (error != nullptr) {
      *error = "bad protocol version";
    }
    return false;
  }

  const std::size_t expected_bytes =
      sizeof(PutPayloadHeader) + header.key_count * sizeof(std::uint64_t) +
      static_cast<std::size_t>(header.key_count) * header.embedding_dim *
          sizeof(float);

  if (payload.size() != expected_bytes) {
    if (error != nullptr) {
      *error = "payload byte size mismatch";
    }
    return false;
  }

  decoded->embedding_dim = header.embedding_dim;
  decoded->keys.resize(header.key_count);
  decoded->values.resize(
      static_cast<std::size_t>(header.key_count) * header.embedding_dim);

  const char* cursor = payload.data() + sizeof(PutPayloadHeader);
  std::memcpy(
      decoded->keys.data(), cursor, header.key_count * sizeof(std::uint64_t));
  cursor += header.key_count * sizeof(std::uint64_t);
  std::memcpy(
      decoded->values.data(), cursor, decoded->values.size() * sizeof(float));

  return true;
}

inline bool EncodePutRemoteControlV2(const PutRemotePayloadV2& control,
                                     std::string* payload,
                                     std::string* error) {
  if (payload == nullptr) {
    if (error != nullptr) {
      *error = "payload is null";
    }
    return false;
  }
  if (control.magic != kPutRemotePayloadMagic) {
    if (error != nullptr) {
      *error = "bad control magic";
    }
    return false;
  }
  if (control.version != kPutProtocolVersionV2) {
    if (error != nullptr) {
      *error = "bad control version";
    }
    return false;
  }
  const std::size_t expected_bytes = PutPayloadBytes(
      control.key_count,
      static_cast<std::size_t>(control.embedding_dim) * sizeof(float));
  if (expected_bytes != control.payload_bytes) {
    if (error != nullptr) {
      *error = "control payload_bytes mismatch";
    }
    return false;
  }
  if (control.transfer_mode != kPutV2TransferModeRead &&
      control.transfer_mode != kPutV2TransferModePush) {
    if (error != nullptr) {
      *error = "bad transfer mode";
    }
    return false;
  }
  payload->resize(sizeof(PutRemotePayloadV2));
  std::memcpy(payload->data(), &control, sizeof(PutRemotePayloadV2));
  return true;
}

inline bool DecodePutRemoteControlV2(
    std::string_view payload, PutRemotePayloadV2* control, std::string* error) {
  if (control == nullptr) {
    if (error != nullptr) {
      *error = "control is null";
    }
    return false;
  }
  if (payload.size() != sizeof(PutRemotePayloadV2)) {
    if (error != nullptr) {
      *error = "control size mismatch";
    }
    return false;
  }
  PutRemotePayloadV2 decoded{};
  std::memcpy(&decoded, payload.data(), sizeof(decoded));
  if (decoded.magic != kPutRemotePayloadMagic) {
    if (error != nullptr) {
      *error = "bad control magic";
    }
    return false;
  }
  if (decoded.version != kPutProtocolVersionV2) {
    if (error != nullptr) {
      *error = "bad control version";
    }
    return false;
  }
  const std::size_t expected_bytes = PutPayloadBytes(
      decoded.key_count,
      static_cast<std::size_t>(decoded.embedding_dim) * sizeof(float));
  if (expected_bytes != decoded.payload_bytes) {
    if (error != nullptr) {
      *error = "control payload_bytes mismatch";
    }
    return false;
  }
  if (decoded.transfer_mode != kPutV2TransferModeRead &&
      decoded.transfer_mode != kPutV2TransferModePush) {
    if (error != nullptr) {
      *error = "bad transfer mode";
    }
    return false;
  }
  *control = decoded;
  return true;
}

inline bool IsPutRemoteControlV2(std::string_view payload) {
  if (payload.size() != sizeof(PutRemotePayloadV2)) {
    return false;
  }
  PutRemotePayloadV2 decoded{};
  std::memcpy(&decoded, payload.data(), sizeof(decoded));
  return decoded.magic == kPutRemotePayloadMagic &&
         decoded.version == kPutProtocolVersionV2;
}

} // namespace petps
