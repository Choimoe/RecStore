#include "ps/local_shm/local_shm_client.h"
#include "ps/local_shm/local_shm_futex.h"
#include "ps/local_shm/local_shm_queue.h"

#include <chrono>
#include <cstdlib>
#include <cstring>
#include <thread>
#include <unistd.h>
#include <vector>

namespace recstore {

namespace {

std::size_t InitTablePayloadBytes(const std::string& table_name) {
  return table_name.size() + sizeof(uint64_t) * 2;
}

std::size_t GetPayloadBytes(std::size_t key_count) {
  return sizeof(uint64_t) * key_count;
}

std::size_t PutPayloadBytes(std::size_t key_count, std::size_t embedding_dim) {
  return sizeof(uint64_t) * key_count +
         sizeof(float) * key_count * embedding_dim;
}

std::size_t UpdateFlatPayloadBytes(const std::string& table_name,
                                   std::size_t key_count,
                                   std::size_t embedding_dim) {
  return table_name.size() + sizeof(uint64_t) * key_count +
         sizeof(float) * key_count * embedding_dim;
}

void MarkError(LocalShmSlotHeader* header, LocalStatusCode code) {
  header->status_code = static_cast<uint32_t>(code);
  header->state.store(static_cast<uint32_t>(LocalSlotState::kError));
}

uint32_t ReadQueueSelector(const json& config, uint32_t fallback) {
  if (config.contains("ready_queue_index")) {
    return config["ready_queue_index"].get<uint32_t>();
  }
  if (config.contains("client_rank")) {
    return config["client_rank"].get<uint32_t>();
  }
  if (config.contains("rank")) {
    return config["rank"].get<uint32_t>();
  }
  if (config.contains("local_rank")) {
    return config["local_rank"].get<uint32_t>();
  }
  if (const char* env = std::getenv("RECSTORE_LOCAL_SHM_READY_QUEUE_INDEX")) {
    return static_cast<uint32_t>(std::strtoul(env, nullptr, 10));
  }
  if (const char* env = std::getenv("LOCAL_RANK")) {
    return static_cast<uint32_t>(std::strtoul(env, nullptr, 10));
  }
  if (const char* env = std::getenv("RANK")) {
    return static_cast<uint32_t>(std::strtoul(env, nullptr, 10));
  }
  return fallback;
}

} // namespace

LocalShmPSClient::LocalShmPSClient(json config) : BasePSClient(config) {
  region_name_ = config.value("region_name", "recstore_local_ps");
  timeout_ms_  = config.value("client_timeout_ms", 30000);
  client_id_   = static_cast<uint32_t>(::getpid());
  if (!region_.Attach(region_name_)) {
    LOG(WARNING) << "LocalShmPSClient failed to attach region " << region_name_;
    return;
  }
  ready_queue_id_ = ResolveReadyQueueId(config);
  LOG(INFO) << "LocalShmPSClient attached region=" << region_name_
            << " pid=" << ::getpid() << " client_id=" << client_id_
            << " timeout_ms=" << timeout_ms_
            << " ready_queue_count=" << region_.ready_queue_count()
            << " ready_queue_id=" << ready_queue_id_;
}

int LocalShmPSClient::GetParameter(const base::ConstArray<uint64_t>& keys,
                                   float* values) {
  return GetParameterFlat(
      keys, values, static_cast<int64_t>(keys.Size()), /*embedding_dim=*/0);
}

int LocalShmPSClient::GetParameterFlat(const base::ConstArray<uint64_t>& keys,
                                       float* values,
                                       int64_t num_rows,
                                       int64_t embedding_dim) {
  if (!region_.IsOpen() || values == nullptr) {
    LOG(ERROR) << "LocalShmPSClient::GetParameter invalid_state"
               << " pid=" << static_cast<int>(::getpid())
               << " backend=local_shm"
               << " region_name=" << region_name_
               << " ready_queue_id=" << ready_queue_id_
               << " key_count=" << num_rows
               << " embedding_dim=" << embedding_dim;
    return -1;
  }
  if (num_rows < 0 || keys.Size() != static_cast<size_t>(num_rows)) {
    LOG(ERROR) << "LocalShmPSClient::GetParameter invalid_row_count"
               << " pid=" << static_cast<int>(::getpid())
               << " backend=local_shm"
               << " region_name=" << region_name_
               << " ready_queue_id=" << ready_queue_id_
               << " key_count=" << keys.Size()
               << " num_rows=" << num_rows
               << " embedding_dim=" << embedding_dim;
    return -1;
  }
  if (embedding_dim < 0) {
    LOG(ERROR) << "LocalShmPSClient::GetParameter invalid_embedding_dim"
               << " pid=" << static_cast<int>(::getpid())
               << " backend=local_shm"
               << " region_name=" << region_name_
               << " ready_queue_id=" << ready_queue_id_
               << " key_count=" << num_rows
               << " embedding_dim=" << embedding_dim;
    return -1;
  }
  const int slot = AcquireSlot();
  if (slot < 0) {
    LOG(ERROR) << "LocalShmPSClient::GetParameter acquire_slot_failed"
               << " pid=" << static_cast<int>(::getpid())
               << " backend=local_shm"
               << " region_name=" << region_name_
               << " ready_queue_id=" << ready_queue_id_
               << " key_count=" << num_rows
               << " embedding_dim=" << embedding_dim;
    return -1;
  }

  auto* header  = region_.slot_header(static_cast<uint32_t>(slot));
  auto* payload = region_.slot_payload(static_cast<uint32_t>(slot));
  const std::size_t input_bytes = GetPayloadBytes(keys.Size());
  if (input_bytes > region_.slot_buffer_bytes()) {
    LOG(ERROR) << "LocalShmPSClient::GetParameter input_too_large"
               << " pid=" << static_cast<int>(::getpid())
               << " backend=local_shm"
               << " region_name=" << region_name_
               << " ready_queue_id=" << ready_queue_id_
               << " key_count=" << num_rows
               << " embedding_dim=" << embedding_dim
               << " slot_id=" << slot
               << " input_bytes=" << input_bytes
               << " slot_buffer_bytes=" << region_.slot_buffer_bytes();
    MarkError(header, LocalStatusCode::kBufferTooSmall);
    ReleaseSlot(static_cast<uint32_t>(slot));
    return -1;
  }

  const uint64_t request_id = NextRequestId();
  header->opcode            = static_cast<uint32_t>(LocalOpcode::kGet);
  header->status_code       = static_cast<uint32_t>(LocalStatusCode::kOk);
  header->client_id         = client_id_;
  header->request_id        = request_id;
  header->client_pid        = static_cast<int64_t>(::getpid());
  header->table_name_len    = 0;
  header->key_count         = static_cast<uint32_t>(num_rows);
  header->embedding_dim     = static_cast<uint32_t>(embedding_dim);
  header->input_bytes       = input_bytes;
  header->output_bytes      = 0;
  header->server_seen_epoch = region_.control()->server_epoch;
  header->error_message_len = 0;

  if (keys.Size() > 0) {
    std::memcpy(payload, keys.Data(), input_bytes);
  }
  header->state.store(static_cast<uint32_t>(LocalSlotState::kReady));
  CHECK(LocalShmQueueEnqueue(region_.ready_queue_header(ready_queue_id_),
                             region_.ready_queue_cells(ready_queue_id_),
                             static_cast<uint32_t>(slot)));
  region_.control()->request_doorbell.fetch_add(1, std::memory_order_release);
  FutexWakeAll(&region_.control()->request_doorbell);

  const bool wait_ok = WaitForSlot(static_cast<uint32_t>(slot), request_id);
  if (!wait_ok ||
      header->status_code != static_cast<uint32_t>(LocalStatusCode::kOk)) {
    ReleaseSlot(static_cast<uint32_t>(slot));
    return -1;
  }

  if (embedding_dim > 0) {
    const std::size_t expected_output_bytes =
        sizeof(float) * static_cast<std::size_t>(num_rows) *
        static_cast<std::size_t>(embedding_dim);
    if (header->output_bytes != expected_output_bytes) {
      LOG(ERROR) << "LocalShmPSClient::GetParameter output_size_mismatch"
                 << " pid=" << static_cast<int>(::getpid())
                 << " backend=local_shm"
                 << " region_name=" << region_name_
                 << " ready_queue_id=" << ready_queue_id_
                 << " key_count=" << num_rows
                 << " embedding_dim=" << embedding_dim
                 << " slot_id=" << slot
                 << " request_id=" << request_id
                 << " expected_output_bytes=" << expected_output_bytes
                 << " actual_output_bytes=" << header->output_bytes
                 << " status_code=" << header->status_code
                 << " server_embedding_dim=" << header->embedding_dim;
      ReleaseSlot(static_cast<uint32_t>(slot));
      return -1;
    }
  }

  if (header->output_bytes > 0) {
    std::memcpy(
        values, payload, static_cast<std::size_t>(header->output_bytes));
  }
  ReleaseSlot(static_cast<uint32_t>(slot));
  return 0;
}

int LocalShmPSClient::PutParameter(
    const base::ConstArray<uint64_t>& keys,
    const std::vector<std::vector<float>>& values) {
  if (!region_.IsOpen()) {
    return -1;
  }
  if (keys.Size() != values.size()) {
    return -1;
  }
  const std::size_t embedding_dim = values.empty() ? 0 : values.front().size();
  for (const auto& row : values) {
    if (row.size() != embedding_dim) {
      return -1;
    }
  }

  const int slot = AcquireSlot();
  if (slot < 0) {
    return -1;
  }
  auto* header  = region_.slot_header(static_cast<uint32_t>(slot));
  auto* payload = region_.slot_payload(static_cast<uint32_t>(slot));
  const std::size_t input_bytes = PutPayloadBytes(keys.Size(), embedding_dim);
  if (input_bytes > region_.slot_buffer_bytes()) {
    MarkError(header, LocalStatusCode::kBufferTooSmall);
    ReleaseSlot(static_cast<uint32_t>(slot));
    return -1;
  }

  const uint64_t request_id = NextRequestId();
  header->opcode            = static_cast<uint32_t>(LocalOpcode::kPut);
  header->status_code       = static_cast<uint32_t>(LocalStatusCode::kOk);
  header->client_id         = client_id_;
  header->request_id        = request_id;
  header->client_pid        = static_cast<int64_t>(::getpid());
  header->table_name_len    = 0;
  header->key_count         = static_cast<uint32_t>(keys.Size());
  header->embedding_dim     = static_cast<uint32_t>(embedding_dim);
  header->input_bytes       = input_bytes;
  header->output_bytes      = 0;
  header->server_seen_epoch = region_.control()->server_epoch;
  header->error_message_len = 0;

  uint8_t* cursor = payload;
  if (keys.Size() > 0) {
    std::memcpy(cursor, keys.Data(), sizeof(uint64_t) * keys.Size());
    cursor += sizeof(uint64_t) * keys.Size();
  }
  for (const auto& row : values) {
    if (!row.empty()) {
      std::memcpy(cursor, row.data(), sizeof(float) * row.size());
      cursor += sizeof(float) * row.size();
    }
  }
  header->state.store(static_cast<uint32_t>(LocalSlotState::kReady));
  CHECK(LocalShmQueueEnqueue(region_.ready_queue_header(ready_queue_id_),
                             region_.ready_queue_cells(ready_queue_id_),
                             static_cast<uint32_t>(slot)));
  region_.control()->request_doorbell.fetch_add(1, std::memory_order_release);
  FutexWakeAll(&region_.control()->request_doorbell);

  const bool ok =
      WaitForSlot(static_cast<uint32_t>(slot), request_id) &&
      header->status_code == static_cast<uint32_t>(LocalStatusCode::kOk);
  ReleaseSlot(static_cast<uint32_t>(slot));
  return ok ? 0 : -1;
}

int LocalShmPSClient::UpdateParameter(
    const std::string& table_name,
    const base::ConstArray<uint64_t>& keys,
    const std::vector<std::vector<float>>* grads) {
  if (grads == nullptr) {
    return -1;
  }
  if (keys.Size() != grads->size()) {
    return -1;
  }
  const std::size_t embedding_dim = grads->empty() ? 0 : grads->front().size();
  std::vector<float> flat;
  flat.reserve(grads->size() * embedding_dim);
  for (const auto& row : *grads) {
    if (row.size() != embedding_dim) {
      return -1;
    }
    flat.insert(flat.end(), row.begin(), row.end());
  }
  return UpdateParameterFlat(
      table_name,
      keys,
      flat.empty() ? nullptr : flat.data(),
      static_cast<int64_t>(grads->size()),
      static_cast<int64_t>(embedding_dim));
}

int LocalShmPSClient::UpdateParameterFlat(
    const std::string& table_name,
    const base::ConstArray<uint64_t>& keys,
    const float* grads,
    int64_t num_rows,
    int64_t embedding_dim) {
  if (!region_.IsOpen() || grads == nullptr || num_rows < 0 ||
      embedding_dim <= 0 || keys.Size() != static_cast<size_t>(num_rows)) {
    return -1;
  }
  const int slot = AcquireSlot();
  if (slot < 0) {
    return -1;
  }

  auto* header  = region_.slot_header(static_cast<uint32_t>(slot));
  auto* payload = region_.slot_payload(static_cast<uint32_t>(slot));
  const std::size_t input_bytes = UpdateFlatPayloadBytes(
      table_name, keys.Size(), static_cast<std::size_t>(embedding_dim));
  if (input_bytes > region_.slot_buffer_bytes()) {
    MarkError(header, LocalStatusCode::kBufferTooSmall);
    ReleaseSlot(static_cast<uint32_t>(slot));
    return -1;
  }

  const uint64_t request_id = NextRequestId();
  header->opcode            = static_cast<uint32_t>(LocalOpcode::kUpdateFlat);
  header->status_code       = static_cast<uint32_t>(LocalStatusCode::kOk);
  header->client_id         = client_id_;
  header->request_id        = request_id;
  header->client_pid        = static_cast<int64_t>(::getpid());
  header->table_name_len    = static_cast<uint32_t>(table_name.size());
  header->key_count         = static_cast<uint32_t>(keys.Size());
  header->embedding_dim     = static_cast<uint32_t>(embedding_dim);
  header->input_bytes       = input_bytes;
  header->output_bytes      = 0;
  header->server_seen_epoch = region_.control()->server_epoch;
  header->error_message_len = 0;

  uint8_t* cursor = payload;
  if (!table_name.empty()) {
    std::memcpy(cursor, table_name.data(), table_name.size());
    cursor += table_name.size();
  }
  if (keys.Size() > 0) {
    std::memcpy(cursor, keys.Data(), sizeof(uint64_t) * keys.Size());
    cursor += sizeof(uint64_t) * keys.Size();
  }
  std::memcpy(cursor,
              grads,
              sizeof(float) * keys.Size() * static_cast<size_t>(embedding_dim));
  header->state.store(static_cast<uint32_t>(LocalSlotState::kReady));
  CHECK(LocalShmQueueEnqueue(region_.ready_queue_header(ready_queue_id_),
                             region_.ready_queue_cells(ready_queue_id_),
                             static_cast<uint32_t>(slot)));
  region_.control()->request_doorbell.fetch_add(1, std::memory_order_release);
  FutexWakeAll(&region_.control()->request_doorbell);

  const bool ok =
      WaitForSlot(static_cast<uint32_t>(slot), request_id) &&
      header->status_code == static_cast<uint32_t>(LocalStatusCode::kOk);
  ReleaseSlot(static_cast<uint32_t>(slot));
  return ok ? 0 : -1;
}

int LocalShmPSClient::InitEmbeddingTable(const std::string& table_name,
                                         const EmbeddingTableConfig& config) {
  if (!region_.IsOpen()) {
    return -1;
  }
  const int slot = AcquireSlot();
  if (slot < 0) {
    return -1;
  }

  auto* header  = region_.slot_header(static_cast<uint32_t>(slot));
  auto* payload = region_.slot_payload(static_cast<uint32_t>(slot));
  const std::size_t input_bytes = InitTablePayloadBytes(table_name);
  if (input_bytes > region_.slot_buffer_bytes()) {
    MarkError(header, LocalStatusCode::kBufferTooSmall);
    ReleaseSlot(static_cast<uint32_t>(slot));
    return -1;
  }

  const uint64_t request_id = NextRequestId();
  header->opcode            = static_cast<uint32_t>(LocalOpcode::kInitTable);
  header->status_code       = static_cast<uint32_t>(LocalStatusCode::kOk);
  header->client_id         = client_id_;
  header->request_id        = request_id;
  header->client_pid        = static_cast<int64_t>(::getpid());
  header->table_name_len    = static_cast<uint32_t>(table_name.size());
  header->key_count         = 0;
  header->embedding_dim     = static_cast<uint32_t>(config.embedding_dim);
  header->input_bytes       = input_bytes;
  header->output_bytes      = 0;
  header->server_seen_epoch = region_.control()->server_epoch;
  header->error_message_len = 0;

  uint8_t* cursor = payload;
  if (!table_name.empty()) {
    std::memcpy(cursor, table_name.data(), table_name.size());
    cursor += table_name.size();
  }
  std::memcpy(cursor, &config.num_embeddings, sizeof(config.num_embeddings));
  cursor += sizeof(config.num_embeddings);
  std::memcpy(cursor, &config.embedding_dim, sizeof(config.embedding_dim));
  header->state.store(static_cast<uint32_t>(LocalSlotState::kReady));
  CHECK(LocalShmQueueEnqueue(region_.ready_queue_header(ready_queue_id_),
                             region_.ready_queue_cells(ready_queue_id_),
                             static_cast<uint32_t>(slot)));
  region_.control()->request_doorbell.fetch_add(1, std::memory_order_release);
  FutexWakeAll(&region_.control()->request_doorbell);

  const bool ok =
      WaitForSlot(static_cast<uint32_t>(slot), request_id) &&
      header->status_code == static_cast<uint32_t>(LocalStatusCode::kOk);
  ReleaseSlot(static_cast<uint32_t>(slot));
  return ok ? 0 : -1;
}

int LocalShmPSClient::AsyncGetParameter(const base::ConstArray<uint64_t>& keys,
                                        float* values) {
  return GetParameter(keys, values);
}

void LocalShmPSClient::Command(PSCommand) {}

uint64_t
LocalShmPSClient::PrefetchParameter(const base::ConstArray<uint64_t>&) {
  return 0;
}

bool LocalShmPSClient::IsPrefetchDone(uint64_t) { return false; }

void LocalShmPSClient::WaitForPrefetch(uint64_t) {}

bool LocalShmPSClient::GetPrefetchResult(uint64_t,
                                         std::vector<std::vector<float>>*) {
  return false;
}

bool LocalShmPSClient::GetPrefetchResultFlat(
    uint64_t, std::vector<float>*, int64_t*, int64_t) {
  return false;
}

int LocalShmPSClient::AcquireSlot() {
  if (!region_.IsOpen()) {
    return -1;
  }
  auto* control      = region_.control();
  auto* queue_header = region_.queue_header(LocalQueueKind::kFree);
  auto* queue_cells  = region_.queue_cells(LocalQueueKind::kFree);
  const auto deadline =
      std::chrono::steady_clock::now() + std::chrono::milliseconds(timeout_ms_);

  while (std::chrono::steady_clock::now() < deadline) {
    uint32_t slot_id = 0;
    if (LocalShmQueueDequeue(queue_header, queue_cells, &slot_id)) {
      auto* header = region_.slot_header(slot_id);
      header->state.store(static_cast<uint32_t>(LocalSlotState::kWriting),
                          std::memory_order_release);
      return static_cast<int>(slot_id);
    }
    const uint32_t observed =
        control->free_doorbell.load(std::memory_order_acquire);
    const auto now = std::chrono::steady_clock::now();
    if (now >= deadline) {
      break;
    }
    const auto remaining =
        std::chrono::duration_cast<std::chrono::milliseconds>(deadline - now);
    FutexWaitUntilValueChange(&control->free_doorbell, observed, remaining);
  }
  return -1;
}

void LocalShmPSClient::ReleaseSlot(uint32_t slot_id) {
  auto* header = region_.slot_header(slot_id);
  header->state.store(
      static_cast<uint32_t>(LocalSlotState::kFree), std::memory_order_release);
  CHECK(LocalShmQueueEnqueue(region_.queue_header(LocalQueueKind::kFree),
                             region_.queue_cells(LocalQueueKind::kFree),
                             slot_id));
  region_.control()->free_doorbell.fetch_add(1, std::memory_order_release);
  FutexWakeAll(&region_.control()->free_doorbell);
}

bool LocalShmPSClient::WaitForSlot(uint32_t slot_id, uint64_t request_id) {
  const auto deadline =
      std::chrono::steady_clock::now() + std::chrono::milliseconds(timeout_ms_);
  auto* header = region_.slot_header(slot_id);
  while (std::chrono::steady_clock::now() < deadline) {
    const uint32_t state = header->state.load();
    if (header->request_id == request_id &&
        (state == static_cast<uint32_t>(LocalSlotState::kDone) ||
         state == static_cast<uint32_t>(LocalSlotState::kError))) {
      return true;
    }

    const uint32_t observed =
        header->completion_doorbell.load(std::memory_order_acquire);
    const auto now = std::chrono::steady_clock::now();
    if (now >= deadline) {
      break;
    }
    const auto remaining =
        std::chrono::duration_cast<std::chrono::milliseconds>(deadline - now);
    FutexWaitUntilValueChange(
        &header->completion_doorbell, observed, remaining);
  }
  LOG(ERROR) << "LocalShmPSClient::WaitForSlot timeout request_id="
             << request_id << " slot_id=" << slot_id
             << " state=" << header->state.load()
             << " status_code=" << header->status_code
             << " output_bytes=" << header->output_bytes
             << " completion_doorbell="
             << header->completion_doorbell.load(std::memory_order_acquire)
             << " server_epoch=" << region_.control()->server_epoch;
  header->status_code = static_cast<uint32_t>(LocalStatusCode::kTimeout);
  return false;
}

uint64_t LocalShmPSClient::NextRequestId() {
  return region_.control()->next_request_id.fetch_add(1);
}

uint32_t LocalShmPSClient::ResolveReadyQueueId(const json& config) const {
  const uint32_t queue_count = region_.ready_queue_count();
  CHECK_GT(queue_count, 0U);
  return ReadQueueSelector(config, client_id_) % queue_count;
}

FACTORY_REGISTER(BasePSClient, local_shm, LocalShmPSClient, json);

} // namespace recstore
