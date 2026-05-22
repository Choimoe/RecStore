#include "petps_client.h"

#include <atomic>
#include <algorithm>
#include <chrono>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <limits>
#include <map>
#include <sstream>
#include <stdexcept>
#include <thread>
#include <vector>

#include "petps_magic.h"
#include "ps/base/Postoffice.h"
#include "ps/base/shard_manager.h"
#include "base/string.h"
#include "rdma_transport_mode.h"

DECLARE_int32(value_size);
DECLARE_int32(max_kv_num_per_request);
DEFINE_int32(numa_id, 0, "");
DEFINE_int32(rdma_server_ready_timeout_sec,
             30,
             "Timeout in seconds waiting for petps server ready key");
DEFINE_int32(rdma_server_ready_poll_ms,
             10,
             "Polling interval in milliseconds for petps server ready key");
DEFINE_uint64(
    rdma_client_receive_arena_bytes,
    256 * 1024 * 1024,
    "Per-process max bytes allocated by PetPSClient::GetReceiveBuffer");
DEFINE_uint64(
    rdma_put_client_send_arena_bytes,
    256 * 1024 * 1024,
    "Per-process max bytes allocated by PetPSClient PUT-v2 payload buffers");
DEFINE_int32(
    rdma_put_protocol_version,
    2,
    "RDMA PUT protocol version: 2=remote-payload(default), 1=legacy-inline");
DEFINE_string(rdma_put_v2_transfer_mode,
              "push",
              "RDMA PUT-v2 payload transfer mode: read|push");
DEFINE_string(rdma_transport_mode,
              "raw_message",
              "RDMA transport mode: raw_message|descriptor_doorbell");
DEFINE_uint64(rdma_put_v2_push_slot_bytes,
              256 * 1024,
              "RDMA PUT-v2(push) per-slot bytes reserved on server DSM");
DEFINE_int32(rdma_put_v2_push_slots_per_client,
             8,
             "RDMA PUT-v2(push) slot count reserved per client node");
DEFINE_uint64(
    rdma_put_v2_push_region_offset,
    64 * 1024 * 1024,
    "RDMA PUT-v2(push) base offset in server DSM for client payload slots");
DEFINE_int32(rdma_wait_spin_loops,
             1000,
             "Busy-spin loops before yielding in PetPSClient::WaitRPCFinish");
DEFINE_int32(rdma_wait_yield_loops,
             2000,
             "Yield loops before sleeping in PetPSClient::WaitRPCFinish");
DEFINE_int32(
    rdma_wait_sleep_us,
    50,
    "Sleep duration in microseconds after spin/yield phases in WaitRPCFinish");
DEFINE_int32(
    rdma_wait_timeout_ms,
    60000,
    "Timeout for waiting a single RDMA RPC completion; <=0 means no timeout");

namespace {
constexpr std::uint64_t kClientDsmSizeBytes        = 1 * 1000 * define::MB;
constexpr std::size_t kClientRdmaThreadBufferBytes = 1 * define::MB;

struct SharedRawVerbsClientTransport {
  std::shared_ptr<petps::RawVerbsTransport> transport;
  std::shared_ptr<std::mutex> mu;
};

SharedRawVerbsClientTransport
GetSharedRawVerbsClientTransport(const petps::RawVerbsConfig& raw_config) {
  static std::mutex shared_mu;
  static std::map<std::pair<int, int>, SharedRawVerbsClientTransport>
      shared_by_endpoint;

  std::lock_guard<std::mutex> guard(shared_mu);
  const auto key = std::make_pair(raw_config.global_id, raw_config.local_lane);
  auto it        = shared_by_endpoint.find(key);
  if (it != shared_by_endpoint.end()) {
    return it->second;
  }

  SharedRawVerbsClientTransport shared;
  shared.transport = std::make_shared<petps::RawVerbsTransport>(raw_config);
  shared.transport->PublishAndConnect();
  shared.mu          = std::make_shared<std::mutex>();
  auto [inserted, _] = shared_by_endpoint.emplace(key, shared);
  return inserted->second;
}

bool ShouldValidateRouting() {
  const char* env = std::getenv("RECSTORE_RDMA_VALIDATE_ROUTING");
  return env != nullptr && std::string(env) != "0";
}

bool ShouldTraceRdmaGet() {
  static const bool enabled = [] {
    const char* env = std::getenv("RECSTORE_RDMA_GET_TRACE");
    return env != nullptr && std::string(env) != "0";
  }();
  return enabled;
}

std::uint64_t RdmaGetTraceInterval() {
  static const std::uint64_t interval = [] {
    const char* env = std::getenv("RECSTORE_RDMA_GET_TRACE_INTERVAL");
    if (env == nullptr) {
      return std::uint64_t{5000};
    }
    const auto parsed =
        static_cast<std::uint64_t>(std::strtoull(env, nullptr, 10));
    return parsed == 0 ? std::uint64_t{5000} : parsed;
  }();
  return interval;
}

std::uint64_t ToNs(std::chrono::steady_clock::duration duration) {
  return static_cast<std::uint64_t>(
      std::chrono::duration_cast<std::chrono::nanoseconds>(duration).count());
}

struct RdmaGetClientTraceStats {
  std::atomic<std::uint64_t> count{0};
  std::atomic<std::uint64_t> prep_ns{0};
  std::atomic<std::uint64_t> register_ns{0};
  std::atomic<std::uint64_t> post_ns{0};
  std::atomic<std::uint64_t> wait_ns{0};
  std::atomic<std::uint64_t> total_ns{0};

  void Add(const char* mode,
           std::uint64_t prep,
           std::uint64_t reg,
           std::uint64_t post,
           std::uint64_t wait,
           std::uint64_t total) {
    prep_ns.fetch_add(prep, std::memory_order_relaxed);
    register_ns.fetch_add(reg, std::memory_order_relaxed);
    post_ns.fetch_add(post, std::memory_order_relaxed);
    wait_ns.fetch_add(wait, std::memory_order_relaxed);
    total_ns.fetch_add(total, std::memory_order_relaxed);
    const auto n        = count.fetch_add(1, std::memory_order_relaxed) + 1;
    const auto interval = RdmaGetTraceInterval();
    if (n % interval != 0) {
      return;
    }
    const double denom = static_cast<double>(n) * 1000.0;
    std::ostringstream out;
    out << "component=rdma_get_trace side=client mode=" << mode
        << " count=" << n
        << " prep_us_avg=" << prep_ns.load(std::memory_order_relaxed) / denom
        << " register_us_avg="
        << register_ns.load(std::memory_order_relaxed) / denom
        << " post_us_avg=" << post_ns.load(std::memory_order_relaxed) / denom
        << " wait_us_avg=" << wait_ns.load(std::memory_order_relaxed) / denom
        << " total_us_avg=" << total_ns.load(std::memory_order_relaxed) / denom;
    LOG(INFO) << out.str();
    std::cerr << out.str() << std::endl;
  }
};

RdmaGetClientTraceStats& RawGetClientTraceStats() {
  static RdmaGetClientTraceStats stats;
  return stats;
}

RdmaGetClientTraceStats& DescriptorGetClientTraceStats() {
  static RdmaGetClientTraceStats stats;
  return stats;
}

std::mutex& RawPutV2ReadSendWindowMutex() {
  static std::mutex mu;
  return mu;
}

int MaxPutKeysPerRpc(int value_size_bytes) {
  const std::size_t usable_payload =
      MESSAGE_SIZE - 40 - sizeof(RawMessage) - sizeof(std::uint32_t);
  if (value_size_bytes <= 0 ||
      usable_payload <= sizeof(petps::PutPayloadHeader)) {
    return 0;
  }
  const std::size_t per_key_bytes =
      sizeof(std::uint64_t) + static_cast<std::size_t>(value_size_bytes);
  return static_cast<int>(
      (usable_payload - sizeof(petps::PutPayloadHeader)) / per_key_bytes);
}

struct alignas(64) PutAckSlot {
  std::atomic<int32_t> status;
  char pad[64 - sizeof(std::atomic<int32_t>)];
};

std::int32_t WaitPutAck(std::atomic<int32_t>* ack) {
  int wait_loops   = 0;
  const auto start = std::chrono::steady_clock::now();
  while (ack->load(std::memory_order_acquire) ==
         static_cast<std::int32_t>(petps::RpcStatus::kPending)) {
    if (wait_loops < FLAGS_rdma_wait_spin_loops) {
      ++wait_loops;
      continue;
    }
    if (wait_loops < FLAGS_rdma_wait_spin_loops + FLAGS_rdma_wait_yield_loops) {
      ++wait_loops;
      std::this_thread::yield();
      continue;
    }
    if (FLAGS_rdma_wait_sleep_us > 0) {
      std::this_thread::sleep_for(
          std::chrono::microseconds(FLAGS_rdma_wait_sleep_us));
    } else {
      std::this_thread::yield();
    }
    if (FLAGS_rdma_wait_timeout_ms > 0) {
      const auto elapsed_ms =
          std::chrono::duration_cast<std::chrono::milliseconds>(
              std::chrono::steady_clock::now() - start)
              .count();
      if (elapsed_ms > FLAGS_rdma_wait_timeout_ms) {
        throw std::runtime_error("PUT ack wait timeout timeout_ms=" +
                                 std::to_string(FLAGS_rdma_wait_timeout_ms));
      }
    }
  }
  return ack->load(std::memory_order_acquire);
}

std::uint64_t
DescriptorSlotOffset(std::uint16_t client_node_id, std::uint64_t slot_id) {
  return FLAGS_rdma_put_v2_push_region_offset +
         (static_cast<std::uint64_t>(client_node_id) *
              static_cast<std::uint64_t>(
                  FLAGS_rdma_put_v2_push_slots_per_client) +
          slot_id) *
             FLAGS_rdma_put_v2_push_slot_bytes;
}

void WaitRawCompletion(petps::RawVerbsTransport* transport) {
  petps::RawVerbsCompletion completion{};
  while (!transport->Poll(&completion, FLAGS_rdma_wait_timeout_ms)) {
    std::this_thread::yield();
  }
}

void PostDescriptorDoorbell(
    petps::RawVerbsTransport* transport,
    const void* descriptor_buffer,
    const petps::RdmaDescriptorRequest& req,
    std::size_t slot_bytes,
    int shard,
    std::uint64_t request_id,
    petps::RdmaDescriptorDoorbellPostDecision decision) {
  const petps::RdmaDescriptorDoorbellPostPlan post_plan =
      petps::GetRdmaDescriptorDoorbellPostPlan();
  if (decision.poll_before_post) {
    WaitRawCompletion(transport);
  }
  if (post_plan.use_write_with_imm) {
    transport->WriteWithImm(
        descriptor_buffer,
        req.descriptor_gaddr,
        slot_bytes,
        static_cast<std::uint32_t>(req.slot_id),
        request_id,
        decision.signal_write_with_imm);
    CHECK(decision.signal_write_with_imm ||
          !post_plan.wait_write_with_imm_completion)
        << "write-with-imm completion wait requires a signaled WR";
    if (post_plan.wait_write_with_imm_completion) {
      WaitRawCompletion(transport);
    }
  } else {
    transport->Write(
        descriptor_buffer,
        req.descriptor_gaddr,
        slot_bytes,
        request_id,
        post_plan.signal_descriptor_write);
    if (post_plan.wait_descriptor_write_completion) {
      WaitRawCompletion(transport);
    }
    CHECK(post_plan.signal_doorbell)
        << "descriptor doorbell completion wait requires a signaled doorbell";
    transport->SendDoorbell(static_cast<std::uint16_t>(shard),
                            static_cast<std::uint32_t>(req.slot_id),
                            request_id);
    if (post_plan.wait_doorbell_completion) {
      WaitRawCompletion(transport);
    }
  }
}
} // namespace

namespace petps {

void PetPSClient::Init() {
  static uint64_t dsm_base_addr = (uint64_t)hugePageAlloc(kClientDsmSizeBytes);

  ClusterInfo cluster;
  cluster.serverNR = XPostoffice::GetInstance()->NumServers();
  cluster.clientNR = XPostoffice::GetInstance()->NumClients();
  DSMConfig config(CacheConfig(), cluster, 0, true);
  config.NIC_name = '0' + FLAGS_numa_id;
  config.dsmSize  = kClientDsmSizeBytes;
  config.baseAddr = (uint64_t)dsm_base_addr;
  dsm_ = DSM::getInstance(config, XPostoffice::GetInstance()->GlobalID());

  CHECK_EQ(dsm_->getMyNodeID(), XPostoffice::GetInstance()->GlobalID())
      << "inconsistent postoffice and wq dsm";

  extern int global_socket_id;
  ::global_socket_id = FLAGS_numa_id;
  LOG(INFO) << "set NUMA ID = " << FLAGS_numa_id;

  RdmaTransportMode mode;
  std::string mode_error;
  CHECK(ParseRdmaTransportMode(FLAGS_rdma_transport_mode, &mode, &mode_error))
      << mode_error;
  transport_mode_ = mode;
  if (transport_mode_ == RdmaTransportMode::kDescriptorDoorbell) {
    CHECK_GT(FLAGS_rdma_put_v2_push_slots_per_client, 0)
        << "descriptor_doorbell requires positive slot count";
    CHECK_GT(FLAGS_rdma_put_v2_push_slot_bytes, 0)
        << "descriptor_doorbell requires positive slot bytes";
  }
}

void PetPSClient::InitDescriptorTransports() {
  CHECK(transport_mode_ == RdmaTransportMode::kDescriptorDoorbell);
  const std::vector<int> thread_ids = GetCurrentThreadServerThreadIDs();
  CHECK(!thread_ids.empty());
  for (int server_thread_id : thread_ids) {
    RawVerbsConfig raw_config;
    raw_config.global_id          = XPostoffice::GetInstance()->GlobalID();
    raw_config.local_lane         = server_thread_id;
    raw_config.remote_lane        = server_thread_id;
    raw_config.num_servers        = XPostoffice::GetInstance()->NumServers();
    raw_config.num_clients        = XPostoffice::GetInstance()->NumClients();
    raw_config.numa_id            = FLAGS_numa_id;
    raw_config.connect_to_servers = true;
    raw_config.connect_to_clients = false;
    raw_config.local_base_addr    = dsm_->get_conf()->baseAddr;
    raw_config.local_region_bytes = dsm_->get_conf()->dsmSize;
    raw_config.allocation_start_offset =
        FLAGS_rdma_client_receive_arena_bytes +
        FLAGS_rdma_put_client_send_arena_bytes;
    const std::uint64_t machine_count = static_cast<std::uint64_t>(
        raw_config.num_servers + raw_config.num_clients);
    raw_config.reserved_region_offset = FLAGS_rdma_put_v2_push_region_offset;
    raw_config.reserved_region_bytes =
        machine_count *
        static_cast<std::uint64_t>(FLAGS_rdma_put_v2_push_slots_per_client) *
        FLAGS_rdma_put_v2_push_slot_bytes;
    auto shared = GetSharedRawVerbsClientTransport(raw_config);
    descriptor_transports_by_thread_[server_thread_id]    = shared.transport;
    descriptor_transport_mus_by_thread_[server_thread_id] = shared.mu;
    if (raw_transport_ == nullptr) {
      raw_transport_    = shared.transport;
      raw_transport_mu_ = shared.mu;
    }
  }
  InitDescriptorSlots();
}

void PetPSClient::SetCurrentThreadServerThreadIDs(std::vector<int> thread_ids) {
  CHECK(!thread_ids.empty());
  std::lock_guard<std::mutex> guard(routing_mu_);
  server_thread_ids_by_thread_[std::this_thread::get_id()] =
      std::move(thread_ids);
}

std::vector<int> PetPSClient::GetCurrentThreadServerThreadIDs() const {
  std::lock_guard<std::mutex> guard(routing_mu_);
  const auto it = server_thread_ids_by_thread_.find(std::this_thread::get_id());
  CHECK(it != server_thread_ids_by_thread_.end())
      << "RDMA client thread was not initialized";
  return it->second;
}

void PetPSClient::InitDescriptorSlots() {
  if (!descriptor_send_slots_.empty()) {
    return;
  }
  const int slot_count = FLAGS_rdma_put_v2_push_slots_per_client;
  CHECK_GT(slot_count, 0) << "descriptor_doorbell requires positive slot count";
  descriptor_send_slots_.resize(static_cast<std::size_t>(slot_count), nullptr);
  descriptor_ack_slots_.resize(static_cast<std::size_t>(slot_count), nullptr);
  descriptor_slot_locks_.reserve(static_cast<std::size_t>(slot_count));
  for (int i = 0; i < slot_count; ++i) {
    descriptor_slot_locks_.push_back(std::make_unique<std::mutex>());
  }

  std::lock_guard<std::mutex> transport_guard(*raw_transport_mu_);
  for (int i = 0; i < slot_count; ++i) {
    descriptor_send_slots_[static_cast<std::size_t>(i)] = static_cast<char*>(
        raw_transport_->AllocateRegistered(FLAGS_rdma_put_v2_push_slot_bytes));
    descriptor_ack_slots_[static_cast<std::size_t>(i)] =
        static_cast<std::atomic<std::int32_t>*>(
            raw_transport_->AllocateRegistered(
                sizeof(std::atomic<std::int32_t>)));
  }
}

RawVerbsTransport*
PetPSClient::DescriptorTransportForThread(int thread_id) const {
  auto it = descriptor_transports_by_thread_.find(thread_id);
  CHECK(it != descriptor_transports_by_thread_.end())
      << "missing descriptor raw transport for server thread_id=" << thread_id;
  return it->second.get();
}

std::mutex*
PetPSClient::DescriptorTransportMutexForThread(int thread_id) const {
  auto it = descriptor_transport_mus_by_thread_.find(thread_id);
  CHECK(it != descriptor_transport_mus_by_thread_.end())
      << "missing descriptor raw transport mutex for server thread_id="
      << thread_id;
  return it->second.get();
}

int PetPSClient::GetParameter(base::ConstArray<uint64_t> keys,
                              std::vector<std::vector<float>>* values) {
  values->clear();
  if (keys.Size() == 0) {
    return 0;
  }

  const int embedding_dim = FLAGS_value_size / sizeof(float);
  auto* flat              = static_cast<float*>(GetReceiveBuffer(
      ResponseBufferBytes(static_cast<std::size_t>(keys.Size()))));
  std::memset(
      flat, 0, ResponseBufferBytes(static_cast<std::size_t>(keys.Size())));
  int rpc_id = GetParameter(keys, flat, false, 0);
  WaitRPCFinish(rpc_id);

  // Check status word at the end of the buffer
  const std::int32_t status =
      *reinterpret_cast<std::int32_t*>(flat + keys.Size() * embedding_dim);
  if (status != static_cast<std::int32_t>(RpcStatus::kOk)) {
    LOG(ERROR) << "GetParameter failed with status: "
               << RpcStatusToString(static_cast<RpcStatus>(status));
    RevokeRPCResource(rpc_id);
    return -1;
  }

  values->reserve(keys.Size());
  for (int i = 0; i < keys.Size(); ++i) {
    std::vector<float> row(embedding_dim);
    std::memcpy(row.data(), flat + i * embedding_dim, FLAGS_value_size);
    values->push_back(std::move(row));
  }
  RevokeRPCResource(rpc_id);
  return 0;
}

void PetPSClient::WaitForServerReady() {
  const std::string key = "petps-server-ready-" + std::to_string(shard_);
  const auto deadline =
      std::chrono::steady_clock::now() +
      std::chrono::seconds(FLAGS_rdma_server_ready_timeout_sec);
  std::string value;
  while (std::chrono::steady_clock::now() < deadline) {
    if (XPostoffice::GetInstance()->MemCachedTryGet(key, &value) &&
        value == "1") {
      return;
    }
    std::this_thread::sleep_for(
        std::chrono::milliseconds(FLAGS_rdma_server_ready_poll_ms));
  }
  throw std::runtime_error(
      "Timed out waiting for RDMA server ready key: " + key);
}

std::vector<int> PetPSClient::GetServerThreadIDs() {
  if (transport_mode_ == RdmaTransportMode::kDescriptorDoorbell) {
    const std::string key = RdmaDescriptorWorkerThreadsKey(shard_);
    const auto deadline =
        std::chrono::steady_clock::now() +
        std::chrono::seconds(FLAGS_rdma_server_ready_timeout_sec);
    std::string value;
    std::string error;
    while (std::chrono::steady_clock::now() < deadline) {
      if (XPostoffice::GetInstance()->MemCachedTryGet(key, &value)) {
        std::vector<int> thread_ids;
        CHECK(DecodeRdmaDescriptorWorkerThreads(
            std::string_view(value.data(), value.size()), &thread_ids, &error))
            << error;
        CHECK(!thread_ids.empty())
            << "descriptor worker thread list is empty for key=" << key;
        const std::uint64_t rotate_seed =
            (static_cast<std::uint64_t>(dsm_->getMyNodeID()) << 16) ^
            static_cast<std::uint64_t>(dsm_->getMyThreadID());
        thread_ids =
            RotateRdmaDescriptorServingThreadIDs(thread_ids, rotate_seed);
        LOG(INFO) << base::SFormat(
            "client{} {}th thread are routed to descriptor PS{} ({})th thread",
            dsm_->getMyNodeID(),
            static_cast<int>(dsm_->getMyThreadID()),
            shard_,
            base::ConstArray<int>(
                thread_ids.data(), static_cast<int>(thread_ids.size()))
                .Debug());
        return thread_ids;
      }
      std::this_thread::sleep_for(
          std::chrono::milliseconds(FLAGS_rdma_server_ready_poll_ms));
    }
    throw std::runtime_error(
        "Timed out waiting for RDMA descriptor worker key: " + key);
  }

  auto m     = RawMessage::get_new_msg();
  m->node_id = dsm_->getMyNodeID();
  m->t_id    = dsm_->getMyThreadID();
  m->type    = GET_SERVER_THREADIDS;
  const auto deadline =
      std::chrono::steady_clock::now() +
      std::chrono::seconds(FLAGS_rdma_server_ready_timeout_sec);
  // RPC to the server ID <shard_>
  dsm_->rpc_call(m, shard_, 0);
  uint64_t wr_id;
  while (std::chrono::steady_clock::now() < deadline) {
    auto recv = dsm_->rpc_fast_wait(&wr_id);
    // RECSTORE_LOG_EVERY_MS(WARNING, 5000)
    //     << "client wait the result of GetServerThreadIDs";
    if (recv) {
      CHECK_EQ(recv->type, RESP_GET_SERVER_THREADIDS);
      Cursor cursor;
      Slice extra_data = recv->get_string(cursor);
      base::ConstArray<int> cores(
          (int*)extra_data.s, extra_data.len / sizeof(int));
      LOG(INFO) << base::SFormat(
          "client{} {}th thread are routed to PS{} ({})th thread",
          m->node_id,
          (int)m->t_id,
          shard_,
          cores.Debug());
      return cores.ToVector();
    }
    std::this_thread::sleep_for(
        std::chrono::milliseconds(FLAGS_rdma_server_ready_poll_ms));
  }
  throw std::runtime_error(
      "Timed out waiting for RDMA server thread ids: shard=" +
      std::to_string(shard_) + " node_id=" + std::to_string(m->node_id) +
      " thread_id=" + std::to_string(m->t_id));
}

std::vector<int> PetPSClient::GetRawServerThreadIDsForCurrentThread() {
  std::call_once(raw_server_thread_ids_once_, [this]() {
    raw_server_thread_ids_ = GetServerThreadIDs();
    CHECK(!raw_server_thread_ids_.empty())
        << "raw server thread list is empty for shard=" << shard_;
    raw_server_thread_ids_.resize(1);
  });

  std::vector<int> thread_ids = raw_server_thread_ids_;
  if (!thread_ids.empty()) {
    const std::uint64_t rotate_seed =
        (static_cast<std::uint64_t>(dsm_->getMyNodeID()) << 16) ^
        static_cast<std::uint64_t>(dsm_->getMyThreadID());
    std::rotate(
        thread_ids.begin(),
        thread_ids.begin() +
            static_cast<std::ptrdiff_t>(rotate_seed % thread_ids.size()),
        thread_ids.end());
  }
  return thread_ids;
}

int PetPSClient::SelectServerThreadID() const {
  const std::vector<int> thread_ids = GetCurrentThreadServerThreadIDs();
  CHECK(!thread_ids.empty());
  const uint32_t slot =
      round_robin_.fetch_add(1, std::memory_order_relaxed) % thread_ids.size();
  int ret = thread_ids[slot];
  return ret;
}

std::atomic<int32_t>* PetPSClient::GetPollSlot(uint64_t rpc_id) const {
  std::lock_guard<std::mutex> guard(rpc_mu_);
  auto it = rpcId2PollMap_.find(rpc_id);
  CHECK(it != rpcId2PollMap_.end()) << "unknown rpc_id=" << rpc_id;
  return it->second;
}

std::size_t PetPSClient::ResponseBufferBytes(std::size_t key_count) const {
  return FixedSlotResponseBytes(key_count, FLAGS_value_size);
}

void* PetPSClient::GetReceiveBuffer(size_t size) {
  static std::atomic<uint64_t> client_memory_offset_acc{0};
  const uint64_t old_offset =
      client_memory_offset_acc.fetch_add(static_cast<uint64_t>(size));
  const uint64_t new_offset = old_offset + static_cast<uint64_t>(size);
  if (new_offset > FLAGS_rdma_client_receive_arena_bytes) {
    throw std::runtime_error(
        "RDMA receive arena exhausted: request=" + std::to_string(size) +
        " old_offset=" + std::to_string(old_offset) +
        " limit=" + std::to_string(FLAGS_rdma_client_receive_arena_bytes));
  }
  GlobalAddress gaddr;
  gaddr.nodeID = dsm_->getMyNodeID();
  gaddr.offset = old_offset;
  return dsm_->addr(gaddr);
}

void* PetPSClient::GetSendBuffer(size_t size) {
  // PUT-v2 in current benchmark path is synchronous (wait for ack before next
  // PUT). Reuse a fixed DSM window instead of monotonically consuming arena.
  if (size > FLAGS_rdma_put_client_send_arena_bytes) {
    throw std::runtime_error(
        "RDMA put send arena exhausted: request=" + std::to_string(size) +
        " old_offset=0" +
        " limit=" + std::to_string(FLAGS_rdma_put_client_send_arena_bytes));
  }
  const uint64_t base_offset = FLAGS_rdma_client_receive_arena_bytes;
  if (base_offset + static_cast<uint64_t>(size) > kClientDsmSizeBytes) {
    throw std::runtime_error(
        "RDMA put send arena exceeds DSM size: request=" +
        std::to_string(size) +
        " old_offset=0 base_offset=" + std::to_string(base_offset) +
        " dsm_size=" + std::to_string(kClientDsmSizeBytes));
  }
  GlobalAddress gaddr;
  gaddr.nodeID = dsm_->getMyNodeID();
  gaddr.offset = base_offset;
  return dsm_->addr(gaddr);
}

// this interface assume all keys with the same embedding dimension
int PetPSClient::GetParameter(base::ConstArray<uint64_t> keys,
                              float* values,
                              bool isAsync,
                              int async_req_id) {
  std::lock_guard<std::mutex> get_guard(get_mu_);
  const bool trace_get   = ShouldTraceRdmaGet();
  const auto trace_start = trace_get ? std::chrono::steady_clock::now()
                                     : std::chrono::steady_clock::time_point{};
  if (transport_mode_ == RdmaTransportMode::kDescriptorDoorbell) {
    return GetParameterDescriptor(keys, values, isAsync);
  }

  thread_local auto m = RawMessage::get_new_msg();
  m->clear();
  GlobalAddress gaddr = dsm_->gaddr(values);
  m->type             = RpcType::GET;
  m->receive_gaddr    = gaddr;

  auto* poll = reinterpret_cast<std::atomic<int32_t>*>(
      reinterpret_cast<char*>(values) + keys.Size() * FLAGS_value_size);
  const uint64_t rpc_id = rpcIDAcc_.fetch_add(1, std::memory_order_relaxed);
  {
    std::lock_guard<std::mutex> guard(rpc_mu_);
    rpcId2PollMap_[rpc_id] = poll;
  }
  poll->store(static_cast<std::int32_t>(RpcStatus::kPending),
              std::memory_order_release);

  if (ShouldValidateRouting()) {
    for (auto each : keys) {
      CHECK_EQ(shard_, ShardManager::KeyPartition(each))
          << each << " not belong to this PS";
    }
  }
  const auto trace_registered =
      trace_get ? std::chrono::steady_clock::now()
                : std::chrono::steady_clock::time_point{};

  dsm_->rpc_call(m,
                 shard_,
                 SelectServerThreadID(),
                 Slice(keys.binary_data(), keys.binary_size()));
  const auto trace_posted = trace_get ? std::chrono::steady_clock::now()
                                      : std::chrono::steady_clock::time_point{};

  if (rpc_id > static_cast<std::uint64_t>(std::numeric_limits<int>::max())) {
    {
      std::lock_guard<std::mutex> guard(rpc_mu_);
      rpcId2PollMap_.erase(rpc_id);
    }
    throw std::runtime_error(
        "rpc_id overflow int range: " + std::to_string(rpc_id));
  }

  if (!isAsync) {
    WaitRPCFinish(static_cast<int>(rpc_id));
  }
  if (trace_get) {
    const auto trace_done = std::chrono::steady_clock::now();
    RawGetClientTraceStats().Add(
        "raw_message",
        ToNs(trace_registered - trace_start),
        0,
        ToNs(trace_posted - trace_registered),
        isAsync ? 0 : ToNs(trace_done - trace_posted),
        ToNs(trace_done - trace_start));
  }
  return static_cast<int>(rpc_id);
}

bool PetPSClient::QueryRPCFinished(int rpc_id) {
  std::atomic<int32_t>* poll = nullptr;
  {
    std::lock_guard<std::mutex> guard(rpc_mu_);
    const auto it = rpcId2PollMap_.find(static_cast<uint64_t>(rpc_id));
    if (it != rpcId2PollMap_.end()) {
      poll = it->second;
    }
  }
  if (poll == nullptr) {
    CHECK(transport_mode_ == RdmaTransportMode::kDescriptorDoorbell)
        << "unknown rpc_id=" << rpc_id;
    return true;
  }
  return poll->load(std::memory_order_acquire) !=
         static_cast<std::int32_t>(RpcStatus::kPending);
}

void PetPSClient::WaitRPCFinish(int rpc_id) {
  std::atomic<int32_t>* poll = nullptr;
  {
    std::lock_guard<std::mutex> guard(rpc_mu_);
    const auto it = rpcId2PollMap_.find(static_cast<uint64_t>(rpc_id));
    if (it != rpcId2PollMap_.end()) {
      poll = it->second;
    }
  }
  if (poll == nullptr) {
    CHECK(transport_mode_ == RdmaTransportMode::kDescriptorDoorbell)
        << "unknown rpc_id=" << rpc_id;
    return;
  }
  int wait_loops   = 0;
  const auto start = std::chrono::steady_clock::now();
  while (poll->load(std::memory_order_acquire) ==
         static_cast<std::int32_t>(RpcStatus::kPending)) {
#ifdef RPC_DEBUG
    RECSTORE_LOG_EVERY_MS(INFO, 1000) << "poll = " << *poll << "\n";
    // << "values[0]=" << values[0] << "\n"
    // << "values[1]=" << values[1] << "\n"
    // << "values[2]=" << values[2] << "\n"
    // << "values[3]=" << values[3] << "\n"
    // << "values[4]=" << values[4] << "\n"
    // << "values[5]=" << values[5] << "\n";
#endif
    if (wait_loops < FLAGS_rdma_wait_spin_loops) {
      ++wait_loops;
      continue;
    }
    if (wait_loops < FLAGS_rdma_wait_spin_loops + FLAGS_rdma_wait_yield_loops) {
      ++wait_loops;
      std::this_thread::yield();
      continue;
    }
    if (FLAGS_rdma_wait_sleep_us > 0) {
      std::this_thread::sleep_for(
          std::chrono::microseconds(FLAGS_rdma_wait_sleep_us));
    } else {
      std::this_thread::yield();
    }
    if (FLAGS_rdma_wait_timeout_ms > 0) {
      const auto elapsed_ms =
          std::chrono::duration_cast<std::chrono::milliseconds>(
              std::chrono::steady_clock::now() - start)
              .count();
      if (elapsed_ms > FLAGS_rdma_wait_timeout_ms) {
        throw std::runtime_error(
            "WaitRPCFinish timeout rpc_id=" + std::to_string(rpc_id) +
            " timeout_ms=" + std::to_string(FLAGS_rdma_wait_timeout_ms));
      }
    }
  }
  return;
}

void PetPSClient::RevokeRPCResource(int rpc_id) {
  std::lock_guard<std::mutex> guard(rpc_mu_);
  const auto rpc_key = static_cast<uint64_t>(rpc_id);
  rpcId2PollMap_.erase(rpc_key);
  descriptor_pending_rpcs_.erase(rpc_key);
};

int PetPSClient::GetParameterDescriptor(
    base::ConstArray<uint64_t> keys, float* values, bool isAsync) {
  const bool trace_get   = ShouldTraceRdmaGet();
  const auto trace_start = trace_get ? std::chrono::steady_clock::now()
                                     : std::chrono::steady_clock::time_point{};
  const RdmaDescriptorClientCompletionMode completion_mode =
      GetRdmaDescriptorGetCompletionMode(isAsync);
  const std::size_t response_bytes =
      FixedSlotResponseBytes(keys.Size(), FLAGS_value_size);
  auto* status = reinterpret_cast<std::atomic<std::int32_t>*>(
      reinterpret_cast<char*>(values) + keys.Size() * FLAGS_value_size);
  status->store(static_cast<std::int32_t>(RpcStatus::kPending),
                std::memory_order_release);

  const std::uint64_t request_id =
      descriptor_request_id_.fetch_add(1, std::memory_order_relaxed);
  if (request_id >
      static_cast<std::uint64_t>(std::numeric_limits<int>::max())) {
    throw std::runtime_error("descriptor request_id overflow int range: " +
                             std::to_string(request_id));
  }
  const std::uint64_t slot_id =
      descriptor_slot_cursor_.fetch_add(1, std::memory_order_relaxed) %
      static_cast<std::uint64_t>(FLAGS_rdma_put_v2_push_slots_per_client);
  std::unique_lock<std::mutex> descriptor_slot_guard(
      *descriptor_slot_locks_[static_cast<std::size_t>(slot_id)]);
  const std::uint16_t client_node_id =
      static_cast<std::uint16_t>(dsm_->getMyNodeID());
  const std::uint64_t descriptor_offset =
      DescriptorSlotOffset(client_node_id, slot_id);
  const int target_thread_id = SelectServerThreadID();
  auto* descriptor_transport = DescriptorTransportForThread(target_thread_id);
  auto* descriptor_transport_mu =
      DescriptorTransportMutexForThread(target_thread_id);

  RdmaDescriptorRequest req{};
  req.magic            = kRdmaDescriptorMagic;
  req.version          = kRdmaDescriptorVersionV1;
  req.op               = static_cast<std::uint16_t>(RdmaDescriptorOp::kGet);
  req.request_id       = request_id;
  req.client_node_id   = client_node_id;
  req.client_thread_id = static_cast<std::uint16_t>(dsm_->getMyThreadID());
  req.lane_id          = static_cast<std::uint32_t>(target_thread_id);
  req.slot_id          = static_cast<std::uint32_t>(slot_id);
  req.key_count        = static_cast<std::uint32_t>(keys.Size());
  req.embedding_dim =
      static_cast<std::uint32_t>(FLAGS_value_size / sizeof(float));
  req.descriptor_gaddr =
      GlobalAddress{static_cast<std::uint16_t>(shard_), descriptor_offset};
  req.response_gaddr = descriptor_transport->LocalAddress(values);
  req.status_gaddr   = descriptor_transport->LocalAddress(status);
  req.payload_bytes  = static_cast<std::uint32_t>(keys.binary_size());
  req.response_bytes = static_cast<std::uint32_t>(response_bytes);

  std::string error;
  const std::size_t slot_bytes =
      sizeof(RdmaDescriptorRequest) + keys.binary_size();
  CHECK_LE(slot_bytes, FLAGS_rdma_put_v2_push_slot_bytes)
      << "descriptor GET request does not fit slot payload_bytes="
      << keys.binary_size()
      << " slot_bytes=" << FLAGS_rdma_put_v2_push_slot_bytes;
  char* descriptor_buffer =
      descriptor_send_slots_[static_cast<std::size_t>(slot_id)];
  CHECK(WriteRdmaDescriptorRequest(
      req, descriptor_buffer, FLAGS_rdma_put_v2_push_slot_bytes, &error))
      << error;
  std::memcpy(descriptor_buffer + sizeof(RdmaDescriptorRequest),
              keys.binary_data(),
              keys.binary_size());
  const auto trace_prepared =
      trace_get ? std::chrono::steady_clock::now()
                : std::chrono::steady_clock::time_point{};

  if (completion_mode == RdmaDescriptorClientCompletionMode::kReturnAfterPost) {
    std::lock_guard<std::mutex> guard(rpc_mu_);
    rpcId2PollMap_[request_id] = status;
    DescriptorPendingRpc pending;
    pending.descriptor_slot_guard = std::move(descriptor_slot_guard);
    descriptor_pending_rpcs_.emplace(request_id, std::move(pending));
  }
  const auto trace_registered =
      trace_get ? std::chrono::steady_clock::now()
                : std::chrono::steady_clock::time_point{};

  try {
    std::lock_guard<std::mutex> transport_guard(*descriptor_transport_mu);
    PostDescriptorDoorbell(
        descriptor_transport,
        descriptor_buffer,
        req,
        slot_bytes,
        shard_,
        request_id,
        descriptor_doorbell_post_state_.Next(target_thread_id));
  } catch (...) {
    {
      std::lock_guard<std::mutex> guard(rpc_mu_);
      rpcId2PollMap_.erase(request_id);
      descriptor_pending_rpcs_.erase(request_id);
    }
    throw;
  }
  const auto trace_posted = trace_get ? std::chrono::steady_clock::now()
                                      : std::chrono::steady_clock::time_point{};
  if (completion_mode ==
      RdmaDescriptorClientCompletionMode::kWaitForCompletion) {
    try {
      WaitPutAck(status);
    } catch (...) {
      std::lock_guard<std::mutex> guard(rpc_mu_);
      rpcId2PollMap_.erase(request_id);
      throw;
    }
  }
  if (trace_get) {
    const auto trace_done = std::chrono::steady_clock::now();
    DescriptorGetClientTraceStats().Add(
        "descriptor_doorbell",
        ToNs(trace_prepared - trace_start),
        ToNs(trace_registered - trace_prepared),
        ToNs(trace_posted - trace_registered),
        completion_mode ==
                RdmaDescriptorClientCompletionMode::kWaitForCompletion
            ? ToNs(trace_done - trace_posted)
            : 0,
        ToNs(trace_done - trace_start));
  }
  return static_cast<int>(request_id);
}

int PetPSClient::PutParameterDescriptor(
    const std::vector<uint64_t>& keys,
    const std::vector<std::vector<float>>& values) {
  const std::uint64_t request_id =
      descriptor_request_id_.fetch_add(1, std::memory_order_relaxed);
  const std::uint64_t slot_count =
      static_cast<std::uint64_t>(FLAGS_rdma_put_v2_push_slots_per_client);
  const std::uint64_t slot_id =
      descriptor_slot_cursor_.fetch_add(1, std::memory_order_relaxed) %
      slot_count;
  std::unique_lock<std::mutex> descriptor_slot_guard(
      *descriptor_slot_locks_[static_cast<std::size_t>(slot_id)]);
  auto* ack = descriptor_ack_slots_[static_cast<std::size_t>(slot_id)];
  ack->store(static_cast<std::int32_t>(RpcStatus::kPending),
             std::memory_order_release);
  const std::uint16_t client_node_id =
      static_cast<std::uint16_t>(dsm_->getMyNodeID());
  const std::uint64_t descriptor_offset =
      DescriptorSlotOffset(client_node_id, slot_id);
  const int target_thread_id = SelectServerThreadID();
  auto* descriptor_transport = DescriptorTransportForThread(target_thread_id);
  auto* descriptor_transport_mu =
      DescriptorTransportMutexForThread(target_thread_id);

  std::string control_payload;
  std::unique_lock<std::mutex> payload_slot_guard;
  bool post_push_payload    = false;
  char* push_payload_buffer = nullptr;
  GlobalAddress push_payload_gaddr;
  std::size_t push_payload_bytes = 0;
  std::unique_lock<std::mutex> read_send_window_guard;
  if (FLAGS_rdma_put_protocol_version == 2) {
    const std::uint32_t transfer_mode = [&]() -> std::uint32_t {
      if (FLAGS_rdma_put_v2_transfer_mode == "read") {
        return kPutV2TransferModeRead;
      }
      if (FLAGS_rdma_put_v2_transfer_mode == "push") {
        return kPutV2TransferModePush;
      }
      return std::numeric_limits<std::uint32_t>::max();
    }();
    if (transfer_mode == std::numeric_limits<std::uint32_t>::max()) {
      LOG(ERROR) << "Unknown rdma_put_v2_transfer_mode="
                 << FLAGS_rdma_put_v2_transfer_mode << ", expected read|push";
      return -1;
    }

    const std::size_t payload_bytes =
        PutPayloadBytes(keys.size(), FLAGS_value_size);
    if (payload_bytes > std::numeric_limits<std::uint32_t>::max()) {
      LOG(ERROR) << "PUT-v2 payload too large, bytes=" << payload_bytes;
      return -1;
    }

    char* payload_buf = nullptr;
    GlobalAddress payload_gaddr;
    if (transfer_mode == kPutV2TransferModeRead) {
      read_send_window_guard =
          std::unique_lock<std::mutex>(RawPutV2ReadSendWindowMutex());
      try {
        payload_buf = static_cast<char*>(GetSendBuffer(payload_bytes));
      } catch (const std::exception& ex) {
        LOG(ERROR) << "PUT-v2(read) allocate payload buffer failed: "
                   << ex.what();
        return -1;
      }
      payload_gaddr = dsm_->gaddr(payload_buf);
    } else {
      if (payload_bytes > kClientRdmaThreadBufferBytes) {
        LOG(ERROR)
            << "PUT-v2(push) payload exceeds thread RDMA buffer: payload_bytes="
            << payload_bytes
            << " thread_buffer_bytes=" << kClientRdmaThreadBufferBytes;
        return -1;
      }
      if (FLAGS_rdma_put_v2_push_slots_per_client <= 1) {
        LOG(ERROR) << "descriptor PUT-v2(push) requires at least 2 slots per "
                      "client to keep descriptor and payload separate";
        return -1;
      }
      const std::uint64_t payload_slot_id = (slot_id + 1) % slot_count;
      payload_slot_guard                  = std::unique_lock<std::mutex>(
          *descriptor_slot_locks_[static_cast<std::size_t>(payload_slot_id)]);
      if (payload_bytes > FLAGS_rdma_put_v2_push_slot_bytes) {
        LOG(ERROR) << "PUT-v2(push) payload larger than slot: payload_bytes="
                   << payload_bytes
                   << " slot_bytes=" << FLAGS_rdma_put_v2_push_slot_bytes;
        return -1;
      }
      payload_buf =
          descriptor_send_slots_[static_cast<std::size_t>(payload_slot_id)];
      payload_gaddr =
          GlobalAddress{static_cast<std::uint16_t>(shard_),
                        DescriptorSlotOffset(client_node_id, payload_slot_id)};
      post_push_payload   = true;
      push_payload_buffer = payload_buf;
      push_payload_gaddr  = payload_gaddr;
      push_payload_bytes  = payload_bytes;
    }

    std::memcpy(payload_buf, keys.data(), keys.size() * sizeof(std::uint64_t));
    char* values_dst = payload_buf + keys.size() * sizeof(std::uint64_t);
    for (std::size_t i = 0; i < values.size(); ++i) {
      std::memcpy(values_dst + i * FLAGS_value_size,
                  values[i].data(),
                  FLAGS_value_size);
    }
    PutRemotePayloadV2 control{
        kPutRemotePayloadMagic,
        kPutProtocolVersionV2,
        0,
        static_cast<std::uint32_t>(keys.size()),
        static_cast<std::uint32_t>(FLAGS_value_size / sizeof(float)),
        payload_gaddr,
        static_cast<std::uint32_t>(payload_bytes),
        transfer_mode,
        0,
    };
    std::string encode_error;
    if (!EncodePutRemoteControlV2(control, &control_payload, &encode_error)) {
      LOG(ERROR) << "PUT-v2 encode control failed: " << encode_error;
      return -1;
    }
  } else {
    control_payload = EncodePutPayload(keys, values);
    if (control_payload.empty()) {
      LOG(ERROR) << "EncodePutPayload failed for descriptor PUT-v1";
      return -1;
    }
  }

  RdmaDescriptorRequest req{};
  req.magic            = kRdmaDescriptorMagic;
  req.version          = kRdmaDescriptorVersionV1;
  req.op               = static_cast<std::uint16_t>(RdmaDescriptorOp::kPut);
  req.request_id       = request_id;
  req.client_node_id   = client_node_id;
  req.client_thread_id = static_cast<std::uint16_t>(dsm_->getMyThreadID());
  req.lane_id          = static_cast<std::uint32_t>(target_thread_id);
  req.slot_id          = static_cast<std::uint32_t>(slot_id);
  req.key_count        = static_cast<std::uint32_t>(keys.size());
  req.embedding_dim =
      static_cast<std::uint32_t>(FLAGS_value_size / sizeof(float));
  req.descriptor_gaddr =
      GlobalAddress{static_cast<std::uint16_t>(shard_), descriptor_offset};
  req.status_gaddr  = descriptor_transport->LocalAddress(ack);
  req.payload_bytes = static_cast<std::uint32_t>(control_payload.size());

  std::string error;
  const std::size_t slot_bytes =
      sizeof(RdmaDescriptorRequest) + control_payload.size();
  CHECK_LE(slot_bytes, FLAGS_rdma_put_v2_push_slot_bytes)
      << "descriptor PUT control does not fit slot payload_bytes="
      << control_payload.size()
      << " slot_bytes=" << FLAGS_rdma_put_v2_push_slot_bytes;
  char* descriptor_buffer =
      descriptor_send_slots_[static_cast<std::size_t>(slot_id)];
  CHECK(WriteRdmaDescriptorRequest(
      req, descriptor_buffer, FLAGS_rdma_put_v2_push_slot_bytes, &error))
      << error;
  std::memcpy(descriptor_buffer + sizeof(RdmaDescriptorRequest),
              control_payload.data(),
              control_payload.size());

  {
    std::lock_guard<std::mutex> transport_guard(*descriptor_transport_mu);
    if (post_push_payload) {
      const RdmaDescriptorPushPayloadPostPlan payload_plan =
          GetRdmaDescriptorPushPayloadPostPlan();
      CHECK(payload_plan.use_raw_write)
          << "descriptor PUT-v2(push) requires raw write payload push";
      descriptor_transport->Write(
          push_payload_buffer,
          push_payload_gaddr,
          push_payload_bytes,
          request_id,
          payload_plan.signal_payload_write);
      CHECK(!payload_plan.wait_payload_write_completion)
          << "descriptor PUT-v2(push) payload wait is intentionally disabled";
    }
    PostDescriptorDoorbell(
        descriptor_transport,
        descriptor_buffer,
        req,
        slot_bytes,
        shard_,
        request_id,
        descriptor_doorbell_post_state_.Next(target_thread_id));
  }
  const std::int32_t status = WaitPutAck(ack);
  return status == static_cast<std::int32_t>(RpcStatus::kOk) ? 0 : -1;
}

int PetPSClient::PutParameter(const std::vector<uint64_t>& keys,
                              const std::vector<std::vector<float>>& values) {
  std::lock_guard<std::mutex> put_guard(put_mu_);
  CHECK_EQ(keys.size(), values.size());
  if (keys.empty()) {
    return 0;
  }

  const std::size_t embedding_dim = FLAGS_value_size / sizeof(float);
  for (const auto& row : values) {
    if (row.size() != embedding_dim) {
      LOG(ERROR) << "PutParameter value size mismatch, expected dim="
                 << embedding_dim << " got=" << row.size();
      return -1;
    }
  }

  if (FLAGS_rdma_put_protocol_version != 1 &&
      FLAGS_rdma_put_protocol_version != 2) {
    LOG(ERROR) << "Unknown rdma_put_protocol_version="
               << FLAGS_rdma_put_protocol_version;
    return -1;
  }

  if (transport_mode_ == RdmaTransportMode::kDescriptorDoorbell) {
    if (FLAGS_rdma_put_protocol_version == 1) {
      const int max_put_keys = MaxPutKeysPerRpc(FLAGS_value_size);
      if (max_put_keys <= 0) {
        LOG(ERROR) << "Invalid max_put_keys computed for value_size="
                   << FLAGS_value_size;
        return -1;
      }
      for (std::size_t offset = 0; offset < keys.size();
           offset += static_cast<std::size_t>(max_put_keys)) {
        const std::size_t end = std::min(
            offset + static_cast<std::size_t>(max_put_keys), keys.size());
        std::vector<uint64_t> key_slice(
            keys.begin() + offset, keys.begin() + end);
        std::vector<std::vector<float>> value_slice(
            values.begin() + offset, values.begin() + end);
        const int rc = PutParameterDescriptor(key_slice, value_slice);
        if (rc != 0) {
          return rc;
        }
      }
      return 0;
    }
    return PutParameterDescriptor(keys, values);
  }

  if (FLAGS_rdma_put_protocol_version == 2) {
    const std::uint32_t transfer_mode = [&]() -> std::uint32_t {
      if (FLAGS_rdma_put_v2_transfer_mode == "read") {
        return kPutV2TransferModeRead;
      }
      if (FLAGS_rdma_put_v2_transfer_mode == "push") {
        return kPutV2TransferModePush;
      }
      return std::numeric_limits<std::uint32_t>::max();
    }();
    if (transfer_mode == std::numeric_limits<std::uint32_t>::max()) {
      LOG(ERROR) << "Unknown rdma_put_v2_transfer_mode="
                 << FLAGS_rdma_put_v2_transfer_mode << ", expected read|push";
      return -1;
    }

    const std::size_t payload_bytes =
        PutPayloadBytes(keys.size(), FLAGS_value_size);
    if (payload_bytes > std::numeric_limits<std::uint32_t>::max()) {
      LOG(ERROR) << "PUT-v2 payload too large, bytes=" << payload_bytes;
      return -1;
    }

    char* payload_buf = nullptr;
    std::unique_lock<std::mutex> read_send_window_guard;
    if (transfer_mode == kPutV2TransferModeRead) {
      read_send_window_guard =
          std::unique_lock<std::mutex>(RawPutV2ReadSendWindowMutex());
      try {
        payload_buf = static_cast<char*>(GetSendBuffer(payload_bytes));
      } catch (const std::exception& ex) {
        LOG(ERROR) << "PUT-v2(read) allocate payload buffer failed: "
                   << ex.what();
        return -1;
      }
    } else if (transfer_mode == kPutV2TransferModePush) {
      if (payload_bytes > kClientRdmaThreadBufferBytes) {
        LOG(ERROR)
            << "PUT-v2(push) payload exceeds thread RDMA buffer: payload_bytes="
            << payload_bytes
            << " thread_buffer_bytes=" << kClientRdmaThreadBufferBytes;
        return -1;
      }
      // dsm_->write[_sync] uses cacheLKey for local SGE. The local source
      // buffer must come from the registered per-thread RDMA cache buffer.
      payload_buf = dsm_->get_rdma_buffer();
    } else {
      LOG(ERROR) << "PUT-v2 unknown transfer mode=" << transfer_mode;
      return -1;
    }

    std::memcpy(payload_buf, keys.data(), keys.size() * sizeof(std::uint64_t));
    char* values_dst = payload_buf + keys.size() * sizeof(std::uint64_t);
    for (std::size_t i = 0; i < values.size(); ++i) {
      std::memcpy(values_dst + i * FLAGS_value_size,
                  values[i].data(),
                  FLAGS_value_size);
    }

    GlobalAddress payload_gaddr;
    if (transfer_mode == kPutV2TransferModeRead) {
      payload_gaddr = dsm_->gaddr(payload_buf);
    }
    if (transfer_mode == kPutV2TransferModePush) {
      if (FLAGS_rdma_put_v2_push_slot_bytes == 0 ||
          FLAGS_rdma_put_v2_push_slots_per_client <= 0) {
        LOG(ERROR) << "PUT-v2(push) invalid slot config: slot_bytes="
                   << FLAGS_rdma_put_v2_push_slot_bytes << " slots_per_client="
                   << FLAGS_rdma_put_v2_push_slots_per_client;
        return -1;
      }
      if (payload_bytes > FLAGS_rdma_put_v2_push_slot_bytes) {
        LOG(ERROR) << "PUT-v2(push) payload larger than slot: payload_bytes="
                   << payload_bytes
                   << " slot_bytes=" << FLAGS_rdma_put_v2_push_slot_bytes;
        return -1;
      }
      thread_local std::uint64_t slot_cursor = 0;
      const std::uint64_t slot_id =
          slot_cursor %
          static_cast<std::uint64_t>(FLAGS_rdma_put_v2_push_slots_per_client);
      ++slot_cursor;
      const std::uint64_t lane =
          static_cast<std::uint64_t>(dsm_->getMyNodeID());
      const std::uint64_t remote_slot =
          lane * static_cast<std::uint64_t>(
                     FLAGS_rdma_put_v2_push_slots_per_client) +
          slot_id;
      const std::uint64_t slot_offset =
          FLAGS_rdma_put_v2_push_region_offset +
          remote_slot * FLAGS_rdma_put_v2_push_slot_bytes;
      payload_gaddr.nodeID = static_cast<std::uint16_t>(shard_);
      payload_gaddr.offset = slot_offset;
      dsm_->write_sync(payload_buf, payload_gaddr, payload_bytes);
    }

    PutRemotePayloadV2 control{
        kPutRemotePayloadMagic,
        kPutProtocolVersionV2,
        0,
        static_cast<std::uint32_t>(keys.size()),
        static_cast<std::uint32_t>(embedding_dim),
        payload_gaddr,
        static_cast<std::uint32_t>(payload_bytes),
        transfer_mode,
        0,
    };
    std::string control_payload;
    std::string encode_error;
    if (!EncodePutRemoteControlV2(control, &control_payload, &encode_error)) {
      LOG(ERROR) << "PUT-v2 encode control failed: " << encode_error;
      return -1;
    }

    constexpr std::size_t kPutAckSlotCount = 4096;
    thread_local PutAckSlot* ack_slots     = nullptr;
    thread_local std::size_t ack_cursor    = 0;
    if (ack_slots == nullptr) {
      ack_slots = static_cast<PutAckSlot*>(
          GetReceiveBuffer(sizeof(PutAckSlot) * kPutAckSlotCount));
    }
    PutAckSlot* slot = ack_slots + (ack_cursor % kPutAckSlotCount);
    ++ack_cursor;
    auto* ack = &slot->status;
    ack->store(static_cast<std::int32_t>(RpcStatus::kPending),
               std::memory_order_release);

    thread_local auto m = RawMessage::get_new_msg();
    m->clear();
    m->type          = RpcType::PUT;
    m->receive_gaddr = dsm_->gaddr(slot);
    dsm_->rpc_call(m,
                   shard_,
                   SelectServerThreadID(),
                   Slice(control_payload.data(), control_payload.size()));
    const std::int32_t status = WaitPutAck(ack);
    if (status != static_cast<std::int32_t>(RpcStatus::kOk)) {
      LOG(ERROR) << "PUT-v2 failed with status="
                 << RpcStatusToString(static_cast<RpcStatus>(status))
                 << " protocol=2 key_count=" << keys.size()
                 << " payload_bytes=" << payload_bytes;
      return -1;
    }
    return 0;
  }

  const int max_put_keys = MaxPutKeysPerRpc(FLAGS_value_size);
  if (max_put_keys <= 0) {
    LOG(ERROR) << "Invalid max_put_keys computed for value_size="
               << FLAGS_value_size;
    return -1;
  }

  for (std::size_t offset = 0; offset < keys.size();
       offset += static_cast<std::size_t>(max_put_keys)) {
    const std::size_t end =
        std::min(offset + static_cast<std::size_t>(max_put_keys), keys.size());
    std::vector<uint64_t> key_slice(keys.begin() + offset, keys.begin() + end);
    std::vector<std::vector<float>> value_slice(
        values.begin() + offset, values.begin() + end);

    std::string payload = EncodePutPayload(key_slice, value_slice);
    if (payload.empty()) {
      LOG(ERROR) << "EncodePutPayload failed for slice [" << offset << ", "
                 << end << "]";
      return -1;
    }

    constexpr std::size_t kPutAckSlotCount = 4096;
    thread_local PutAckSlot* ack_slots     = nullptr;
    thread_local std::size_t ack_cursor    = 0;
    if (ack_slots == nullptr) {
      ack_slots = static_cast<PutAckSlot*>(
          GetReceiveBuffer(sizeof(PutAckSlot) * kPutAckSlotCount));
    }
    PutAckSlot* slot = ack_slots + (ack_cursor % kPutAckSlotCount);
    ++ack_cursor;
    auto* ack = &slot->status;
    ack->store(static_cast<std::int32_t>(RpcStatus::kPending),
               std::memory_order_release);

    thread_local auto m = RawMessage::get_new_msg();
    m->clear();
    m->type          = RpcType::PUT;
    m->receive_gaddr = dsm_->gaddr(slot);
    dsm_->rpc_call(m,
                   shard_,
                   SelectServerThreadID(),
                   Slice(payload.data(), payload.size()));
    const std::int32_t status = WaitPutAck(ack);
    if (status != static_cast<std::int32_t>(RpcStatus::kOk)) {
      LOG(ERROR) << "PUT-v1 slice failed with status="
                 << RpcStatusToString(static_cast<RpcStatus>(status))
                 << " protocol=1 slice=[" << offset << ", " << end << ")";
      return -1;
    }
  }
  return 0;
}

int PetPSClient::FakePutParameter(base::ConstArray<uint64_t> keys,
                                  float* values) {
  LOG(WARNING) << "FakePutParameter is a benchmark-only path and does not "
                  "carry full put payload values";
  thread_local auto m = RawMessage::get_new_msg();
  m->clear();
  GlobalAddress gaddr = dsm_->gaddr(values);
  m->type             = RpcType::PUT;
  m->receive_gaddr    = gaddr;

  // LOG(INFO) << "send PS Put";
  auto* poll            = reinterpret_cast<std::atomic<int32_t>*>(values);
  const uint64_t rpc_id = rpcIDAcc_.fetch_add(1, std::memory_order_relaxed);
  {
    std::lock_guard<std::mutex> guard(rpc_mu_);
    rpcId2PollMap_[rpc_id] = poll;
  }
  poll->store(static_cast<std::int32_t>(RpcStatus::kPending),
              std::memory_order_release);

  if (ShouldValidateRouting()) {
    for (auto each : keys) {
      CHECK_EQ(shard_, ShardManager::KeyPartition(each))
          << each << " not belong to this PS";
    }
  }

  dsm_->rpc_call(m,
                 shard_,
                 SelectServerThreadID(),
                 Slice(keys.binary_data(), keys.binary_size()));

  if (rpc_id > static_cast<std::uint64_t>(std::numeric_limits<int>::max())) {
    {
      std::lock_guard<std::mutex> guard(rpc_mu_);
      rpcId2PollMap_.erase(rpc_id);
    }
    throw std::runtime_error(
        "rpc_id overflow int range: " + std::to_string(rpc_id));
  }

  // if (!isAsync) {
  //   WaitRPCFinish(rpcIDAcc_);
  // }
  return static_cast<int>(rpc_id);
}

} // namespace petps
