#include <folly/init/Init.h>

#include <atomic>
#include <chrono>
#include <condition_variable>
#include <cstdint>
#include <cstdlib>
#include <deque>
#include <fstream>
#include <future>
#include <iostream>
#include <memory>
#include <mutex>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

#include "base/base.h"
#include "base/factory.h"
#include "base/init.h"
#include "base/log.h"
#include "base/string.h"
#include "base/timer.h"
#include "ps/base/Postoffice.h"
#include "ps/base/base_ps_server.h"
#include "ps/base/cache_ps_impl.h"
#include "ps/base/parameters.h"
#include "ps/base/shard_manager.h"
#include "mayfly_config.h"
#include "memory/epoch_manager.h"
#include "memory/shm_file.h"
#include "ps/rdma/raw_verbs_transport.h"
#include "ps/rdma/rdma_protocol.h"
#include "ps/rdma/rdma_status.h"
#include "ps/rdma/rdma_transport_mode.h"
#include "petps_magic.h"
#include "recstore_config.h"
#include "src/base/config.h"
#include "third_party/Mayfly-main/include/DSM.h"
#include "third_party/json/single_include/nlohmann/json.hpp"

DEFINE_string(config_path, "", "config file path");

DEFINE_double(warmup_ratio,
              0.8,
              "bulk load (warmup_ratio * key_space) kvs in DB");

DEFINE_int32(thread_num, 1, "");
DEFINE_bool(use_sglist, true, "");
DEFINE_bool(preload, false, "");
DEFINE_bool(use_dram, false, "");
DEFINE_int32(numa_id, 0, "");
DEFINE_uint64(rdma_per_thread_response_limit_bytes,
              1 * 1024 * 1024,
              "Per-thread max response bytes for RDMA GET replies");
DEFINE_uint64(rdma_put_server_scratch_bytes,
              1 * 1024 * 1024,
              "Per-thread max bytes used for RDMA PUT-v2 payload scratch");
DEFINE_uint64(rdma_put_v2_push_slot_bytes,
              256 * 1024,
              "RDMA PUT-v2(push) per-slot bytes reserved in server DSM");
DEFINE_int32(rdma_put_v2_push_slots_per_client,
             8,
             "RDMA PUT-v2(push) slot count reserved for each client node");
DEFINE_uint64(
    rdma_put_v2_push_region_offset,
    64 * 1024 * 1024,
    "RDMA PUT-v2(push) base offset in server DSM used by client payload slots");
DEFINE_string(rdma_transport_mode,
              "raw_message",
              "RDMA transport mode: raw_message|descriptor_doorbell");
DEFINE_int32(
    rdma_wait_timeout_ms,
    60000,
    "Timeout for waiting a single raw verbs completion; <=0 means no timeout");

DECLARE_int32(value_size);
DECLARE_int32(max_kv_num_per_request);

namespace {
constexpr std::size_t kRdmaThreadBufferBytes = 1 * define::MB;

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

struct RdmaGetServerHandleTraceStats {
  std::atomic<std::uint64_t> count{0};
  std::atomic<std::uint64_t> parse_ns{0};
  std::atomic<std::uint64_t> index_ns{0};
  std::atomic<std::uint64_t> response_ns{0};
  std::atomic<std::uint64_t> write_ns{0};
  std::atomic<std::uint64_t> total_ns{0};

  void Add(const char* mode,
           std::uint64_t parse,
           std::uint64_t index,
           std::uint64_t response,
           std::uint64_t write,
           std::uint64_t total) {
    parse_ns.fetch_add(parse, std::memory_order_relaxed);
    index_ns.fetch_add(index, std::memory_order_relaxed);
    response_ns.fetch_add(response, std::memory_order_relaxed);
    write_ns.fetch_add(write, std::memory_order_relaxed);
    total_ns.fetch_add(total, std::memory_order_relaxed);
    const auto n        = count.fetch_add(1, std::memory_order_relaxed) + 1;
    const auto interval = RdmaGetTraceInterval();
    if (n % interval != 0) {
      return;
    }
    const double denom = static_cast<double>(n) * 1000.0;
    std::ostringstream out;
    out << "component=rdma_get_trace side=server stage=handle mode=" << mode
        << " count=" << n
        << " parse_us_avg=" << parse_ns.load(std::memory_order_relaxed) / denom
        << " index_us_avg=" << index_ns.load(std::memory_order_relaxed) / denom
        << " response_us_avg="
        << response_ns.load(std::memory_order_relaxed) / denom
        << " write_us_avg=" << write_ns.load(std::memory_order_relaxed) / denom
        << " total_us_avg=" << total_ns.load(std::memory_order_relaxed) / denom;
    LOG(INFO) << out.str();
    std::cerr << out.str() << std::endl;
  }
};

struct RdmaGetServerEnvelopeTraceStats {
  std::atomic<std::uint64_t> count{0};
  std::atomic<std::uint64_t> poll_ns{0};
  std::atomic<std::uint64_t> decode_ns{0};
  std::atomic<std::uint64_t> handle_ns{0};
  std::atomic<std::uint64_t> total_ns{0};

  void Add(std::uint64_t poll,
           std::uint64_t decode,
           std::uint64_t handle,
           std::uint64_t total) {
    poll_ns.fetch_add(poll, std::memory_order_relaxed);
    decode_ns.fetch_add(decode, std::memory_order_relaxed);
    handle_ns.fetch_add(handle, std::memory_order_relaxed);
    total_ns.fetch_add(total, std::memory_order_relaxed);
    const auto n        = count.fetch_add(1, std::memory_order_relaxed) + 1;
    const auto interval = RdmaGetTraceInterval();
    if (n % interval != 0) {
      return;
    }
    const double denom = static_cast<double>(n) * 1000.0;
    std::ostringstream out;
    out << "component=rdma_get_trace side=server stage=envelope mode="
        << "descriptor_doorbell count=" << n << " poll_us_avg="
        << poll_ns.load(std::memory_order_relaxed) / denom << " decode_us_avg="
        << decode_ns.load(std::memory_order_relaxed) / denom
        << " handle_us_avg="
        << handle_ns.load(std::memory_order_relaxed) / denom
        << " total_us_avg=" << total_ns.load(std::memory_order_relaxed) / denom;
    LOG(INFO) << out.str();
    std::cerr << out.str() << std::endl;
  }
};

RdmaGetServerHandleTraceStats& RawGetServerHandleTraceStats() {
  static RdmaGetServerHandleTraceStats stats;
  return stats;
}

RdmaGetServerHandleTraceStats& DescriptorGetServerHandleTraceStats() {
  static RdmaGetServerHandleTraceStats stats;
  return stats;
}

RdmaGetServerEnvelopeTraceStats& DescriptorGetServerEnvelopeTraceStats() {
  static RdmaGetServerEnvelopeTraceStats stats;
  return stats;
}

struct RequestView {
  RpcType type;
  NodeIDType node_id;
  ThreadIDType t_id;
  GlobalAddress receive_gaddr;
  Slice payload;
  bool descriptor_doorbell;
};

RequestView RequestViewFromRawMessage(RawMessage* recv) {
  Cursor cursor;
  return RequestView{
      recv->type,
      recv->node_id,
      recv->t_id,
      recv->receive_gaddr,
      recv->get_string(cursor),
      false,
  };
}
} // namespace

namespace recstore {
class PetPSServer : public BaseParameterServer {
public:
  PetPSServer(CachePS* cache_ps, int thread_count)
      : cache_ps_(cache_ps),
        thread_count_(thread_count),
        get_parameter_timer_("GetParameter", 1),
        index_timer_("Index Part", 1),
        value_timer_("Value Part", 1),
        epoch_manager_(base::epoch::EpochManager::GetInstance()) {
    CHECK_LE(thread_count, kMaxThread);

    ClusterInfo cluster;
    cluster.serverNR = XPostoffice::GetInstance()->NumServers();
    cluster.clientNR = XPostoffice::GetInstance()->NumClients();

    DSMConfig config(CacheConfig(), cluster, 0, false);
    if (FLAGS_use_sglist) {
      // TODO: Need to implement PM address registration for cache_ps
      LOG(WARNING) << "PM address registration not implemented for cache_ps, "
                      "using default DRAM allocation";
      config.dsmSize  = 100 * define::MB;
      config.baseAddr = (uint64_t)hugePageAlloc(config.dsmSize);
      LOG(INFO) << "Using DRAM space instead of PM space";
    } else {
      config.dsmSize  = 100 * define::MB;
      config.baseAddr = (uint64_t)hugePageAlloc(config.dsmSize);
      LOG(INFO) << "WE DONT register PM space to RNIC";
    }
    LOG(INFO) << "register MR start =" << (void*)config.baseAddr
              << ", end = " << (void*)(config.baseAddr + config.dsmSize)
              << ", size = " << config.dsmSize;

    config.NIC_name = '0' + FLAGS_numa_id;
    {
      dsm_ = DSM::getInstance(config, XPostoffice::GetInstance()->GlobalID());
      CHECK_EQ(dsm_->getMyNodeID(), XPostoffice::GetInstance()->GlobalID())
          << "inconsistent postoffice and wq dsm";
      LOG(INFO) << "xmh: finish construct DSM";
    }
    sourcelists_.resize(thread_count);
    for (int i = 0; i < thread_count; i++) {
      sourcelists_[i].resize(FLAGS_max_kv_num_per_request);
    }
    std::string mode_error;
    CHECK(petps::ParseRdmaTransportMode(
        FLAGS_rdma_transport_mode, &transport_mode_, &mode_error))
        << mode_error;
    if (transport_mode_ == petps::RdmaTransportMode::kDescriptorDoorbell) {
      CHECK_GT(FLAGS_rdma_put_v2_push_slots_per_client, 0)
          << "descriptor_doorbell requires positive slot count";
      CHECK_GT(FLAGS_rdma_put_v2_push_slot_bytes, 0)
          << "descriptor_doorbell requires positive slot bytes";
      raw_transports_.resize(static_cast<std::size_t>(thread_count_));
      const std::uint64_t machine_count = static_cast<std::uint64_t>(
          XPostoffice::GetInstance()->NumServers() +
          XPostoffice::GetInstance()->NumClients());
      for (int thread_id :
           petps::GetRdmaDescriptorServingThreadIDs(thread_count_)) {
        petps::RawVerbsConfig raw_config;
        raw_config.global_id   = XPostoffice::GetInstance()->GlobalID();
        raw_config.local_lane  = thread_id;
        raw_config.remote_lane = thread_id;
        raw_config.num_servers = XPostoffice::GetInstance()->NumServers();
        raw_config.num_clients = XPostoffice::GetInstance()->NumClients();
        raw_config.numa_id     = FLAGS_numa_id;
        raw_config.connect_to_servers = false;
        raw_config.connect_to_clients = true;
        raw_config.local_base_addr    = dsm_->get_conf()->baseAddr;
        raw_config.local_region_bytes = dsm_->get_conf()->dsmSize;
        raw_config.reserved_region_offset =
            FLAGS_rdma_put_v2_push_region_offset;
        raw_config.reserved_region_bytes =
            machine_count *
            static_cast<std::uint64_t>(
                FLAGS_rdma_put_v2_push_slots_per_client) *
            FLAGS_rdma_put_v2_push_slot_bytes;
        raw_transports_[static_cast<std::size_t>(thread_id)] =
            std::make_unique<petps::RawVerbsTransport>(raw_config);
      }
    }
  }

  void Run() {
    for (int i = 0; i < thread_count_; i++) {
      LOG(INFO) << "Starts PS polling thread " << i;
      threads_.emplace_back(&PetPSServer::PollingThread, this, i);
      tp[i][0] = 0;
    }
  }

  uint64_t GetThroughputCounterSum() const {
    uint64_t sum = 0;
    for (int i = 0; i < thread_count_; i++)
      sum += tp[i][0];
    return sum;
  }

private:
  struct DescriptorTask {
    petps::RdmaDescriptorRequest request;
    const char* inline_payload       = nullptr;
    std::size_t inline_payload_bytes = 0;
  };

  void WriteMayflyResponse(
      const RequestView& request,
      const char* buffer,
      GlobalAddress gaddr,
      std::size_t size,
      bool signal,
      std::uint64_t wr_id,
      int thread_id) {
    if (!request.descriptor_doorbell) {
      if (signal) {
        dsm_->write_sync(buffer, gaddr, size);
      } else {
        dsm_->write(buffer, gaddr, size, signal, wr_id);
      }
      return;
    }

    auto& counter = descriptor_dsm_write_counters_[thread_id];
    const petps::RdmaDescriptorDsmWriteDecision decision =
        petps::GetRdmaDescriptorDsmWriteDecision(counter);
    ++counter;
    if (decision.poll_before_write) {
      dsm_->poll_rdma_cq(1);
    }
    dsm_->write(buffer, gaddr, size, decision.signal_write, wr_id);
  }

  void RpcGetServerServingThreadIDs(RawMessage* recv) {
    CHECK_EQ(recv->type, GET_SERVER_THREADIDS);
    VLOG(1) << "component=rdma_server event=get_server_threadids_recv node_id="
            << static_cast<int>(recv->node_id)
            << " tid=" << static_cast<int>(recv->t_id);
    static std::atomic_int serving_thread_id{0};
    auto m  = RawMessage::get_new_msg();
    m->type = RESP_GET_SERVER_THREADIDS;
    std::vector<int> thread_ids;
    thread_ids.reserve(static_cast<std::size_t>(thread_count_));
    const int start = serving_thread_id.fetch_add(1);
    for (int i = 0; i < thread_count_; ++i) {
      thread_ids.push_back((start + i) % thread_count_);
    }
    dsm_->rpc_call(
        m,
        recv->node_id,
        recv->t_id,
        Slice((char*)thread_ids.data(), thread_ids.size() * sizeof(int)));
    VLOG(1) << "component=rdma_server event=get_server_threadids_reply node_id="
            << static_cast<int>(recv->node_id) << " tid="
            << static_cast<int>(recv->t_id) << " threads=" << thread_ids.size();
  }

  void PublishDescriptorWorkerThreads() {
    if (transport_mode_ == petps::RdmaTransportMode::kDescriptorDoorbell) {
      const std::vector<int> thread_ids =
          petps::GetRdmaDescriptorServingThreadIDs(thread_count_);
      const std::string worker_key = petps::RdmaDescriptorWorkerThreadsKey(
          XPostoffice::GetInstance()->ServerID());
      XPostoffice::GetInstance()->MemCachedSet(
          worker_key, petps::EncodeRdmaDescriptorWorkerThreads(thread_ids));
      VLOG(1) << "component=rdma_server "
              << "event=publish_descriptor_worker_threads key=" << worker_key
              << " threads=" << thread_ids.size();
    }
  }

  void PublishReadyKeys() {
    if (ready_published_.exchange(true, std::memory_order_acq_rel)) {
      return;
    }
    PublishDescriptorWorkerThreads();
    const std::string ready_key =
        "petps-server-ready-" +
        std::to_string(XPostoffice::GetInstance()->ServerID());
    XPostoffice::GetInstance()->MemCachedSet(ready_key, "1");
    VLOG(1) << "component=rdma_server event=publish_ready_key key="
            << ready_key;
  }

  void HandlePsPut(const RequestView& request, int thread_id) {
    std::string error;
    petps::RpcStatus status = petps::RpcStatus::kOk;
    const std::string_view payload_view(request.payload.s, request.payload.len);

    petps::PutRemotePayloadV2 control{};
    if (petps::DecodePutRemoteControlV2(payload_view, &control, &error)) {
      if (control.embedding_dim * sizeof(float) != FLAGS_value_size) {
        LOG(ERROR) << "RpcPsPut(v2) value size mismatch, embedding_dim="
                   << control.embedding_dim
                   << " FLAGS_value_size=" << FLAGS_value_size;
        status = petps::RpcStatus::kValueSizeMismatch;
      } else if (control.key_count >
                 static_cast<std::uint32_t>(FLAGS_max_kv_num_per_request)) {
        LOG(ERROR) << "RpcPsPut(v2) batch too large, key_count="
                   << control.key_count << " max_kv_num_per_request="
                   << FLAGS_max_kv_num_per_request;
        status = petps::RpcStatus::kBatchTooLarge;
      } else {
        const char* payload_ptr = nullptr;
        if (control.transfer_mode == petps::kPutV2TransferModeRead) {
          const std::size_t scratch_limit = std::min<std::size_t>(
              FLAGS_rdma_put_server_scratch_bytes, kRdmaThreadBufferBytes);
          if (control.payload_bytes > scratch_limit) {
            LOG(ERROR) << "RpcPsPut(v2-read) scratch overflow, payload_bytes="
                       << control.payload_bytes
                       << " scratch_limit=" << scratch_limit;
            status = petps::RpcStatus::kBatchTooLarge;
          } else {
            char* put_scratch = dsm_->get_rdma_buffer();
            dsm_->read_sync(
                put_scratch, control.payload_gaddr, control.payload_bytes);
            payload_ptr = put_scratch;
          }
        } else if (control.transfer_mode == petps::kPutV2TransferModePush) {
          if (FLAGS_rdma_put_v2_push_slot_bytes == 0 ||
              FLAGS_rdma_put_v2_push_slots_per_client <= 0) {
            LOG(ERROR) << "RpcPsPut(v2-push) invalid slot config: slot_bytes="
                       << FLAGS_rdma_put_v2_push_slot_bytes
                       << " slots_per_client="
                       << FLAGS_rdma_put_v2_push_slots_per_client;
            status = petps::RpcStatus::kInvalidPayload;
          } else if (control.payload_bytes >
                     FLAGS_rdma_put_v2_push_slot_bytes) {
            LOG(ERROR)
                << "RpcPsPut(v2-push) payload larger than slot: payload_bytes="
                << control.payload_bytes
                << " slot_bytes=" << FLAGS_rdma_put_v2_push_slot_bytes;
            status = petps::RpcStatus::kBatchTooLarge;
          } else if (control.payload_gaddr.nodeID != dsm_->getMyNodeID()) {
            LOG(ERROR)
                << "RpcPsPut(v2-push) payload node mismatch: payload_node="
                << control.payload_gaddr.nodeID
                << " local_node=" << dsm_->getMyNodeID();
            status = petps::RpcStatus::kInvalidPayload;
          } else {
            const std::uint64_t slots_per_client = static_cast<std::uint64_t>(
                FLAGS_rdma_put_v2_push_slots_per_client);
            const std::uint64_t total_slots =
                static_cast<std::uint64_t>(dsm_->get_conf()->machineNR) *
                slots_per_client;
            const std::uint64_t region_bytes =
                total_slots * FLAGS_rdma_put_v2_push_slot_bytes;
            const std::uint64_t region_begin =
                FLAGS_rdma_put_v2_push_region_offset;
            const std::uint64_t region_end    = region_begin + region_bytes;
            const std::uint64_t payload_begin = control.payload_gaddr.offset;
            const std::uint64_t payload_end =
                payload_begin +
                static_cast<std::uint64_t>(control.payload_bytes);
            const std::uint64_t sender_lane_begin =
                region_begin +
                static_cast<std::uint64_t>(request.node_id) * slots_per_client *
                    FLAGS_rdma_put_v2_push_slot_bytes;
            const std::uint64_t sender_lane_end =
                sender_lane_begin +
                slots_per_client * FLAGS_rdma_put_v2_push_slot_bytes;

            if (region_end <= region_begin) {
              LOG(ERROR) << "RpcPsPut(v2-push) invalid region range begin="
                         << region_begin << " end=" << region_end;
              status = petps::RpcStatus::kInvalidPayload;
            } else if (payload_begin < region_begin ||
                       payload_end > region_end ||
                       payload_end < payload_begin) {
              LOG(ERROR) << "RpcPsPut(v2-push) payload out of range: payload=["
                         << payload_begin << "," << payload_end << ") region=["
                         << region_begin << "," << region_end << ")";
              status = petps::RpcStatus::kInvalidPayload;
            } else if ((payload_begin - region_begin) %
                           FLAGS_rdma_put_v2_push_slot_bytes !=
                       0) {
              LOG(ERROR) << "RpcPsPut(v2-push) payload not slot-aligned: "
                            "payload_begin="
                         << payload_begin << " region_begin=" << region_begin
                         << " slot_bytes=" << FLAGS_rdma_put_v2_push_slot_bytes;
              status = petps::RpcStatus::kInvalidPayload;
            } else if (payload_begin < sender_lane_begin ||
                       payload_end > sender_lane_end ||
                       sender_lane_end < sender_lane_begin) {
              LOG(ERROR)
                  << "RpcPsPut(v2-push) payload not in sender lane: sender="
                  << static_cast<int>(request.node_id) << " payload=["
                  << payload_begin << "," << payload_end << ") sender_lane=["
                  << sender_lane_begin << "," << sender_lane_end << ")";
              status = petps::RpcStatus::kInvalidPayload;
            } else if (payload_end >
                       static_cast<std::uint64_t>(dsm_->get_conf()->dsmSize)) {
              LOG(ERROR)
                  << "RpcPsPut(v2-push) payload exceeds DSM size: payload_end="
                  << payload_end << " dsm_size=" << dsm_->get_conf()->dsmSize;
              status = petps::RpcStatus::kInvalidPayload;
            } else {
              payload_ptr = dsm_->addr(control.payload_gaddr);
            }
          }
        } else {
          LOG(ERROR) << "RpcPsPut(v2) unknown transfer_mode="
                     << control.transfer_mode;
          status = petps::RpcStatus::kInvalidPayload;
        }

        if (status == petps::RpcStatus::kOk && payload_ptr != nullptr) {
          const auto* keys =
              reinterpret_cast<const std::uint64_t*>(payload_ptr);
          const char* value_bytes =
              payload_ptr + static_cast<std::size_t>(control.key_count) *
                                sizeof(std::uint64_t);
          cache_ps_->PutDenseParameterBatch(
              keys,
              reinterpret_cast<const float*>(value_bytes),
              static_cast<int>(control.key_count),
              static_cast<int>(control.embedding_dim),
              thread_id);
        } else if (status == petps::RpcStatus::kOk) {
          status = petps::RpcStatus::kInvalidPayload;
        }
      }
    } else {
      petps::DecodedPutPayload decoded;
      if (!petps::DecodePutPayload(payload_view, &decoded, &error)) {
        LOG(ERROR) << "RpcPsPut decode error: " << error;
        status = petps::RpcStatus::kInvalidPayload;
      } else if (decoded.embedding_dim * sizeof(float) != FLAGS_value_size) {
        LOG(ERROR) << "RpcPsPut(v1) value size mismatch, embedding_dim="
                   << decoded.embedding_dim
                   << " FLAGS_value_size=" << FLAGS_value_size;
        status = petps::RpcStatus::kValueSizeMismatch;
      } else if (decoded.keys.size() >
                 static_cast<std::size_t>(FLAGS_max_kv_num_per_request)) {
        LOG(ERROR) << "RpcPsPut(v1) batch too large, key_count="
                   << decoded.keys.size() << " max_kv_num_per_request="
                   << FLAGS_max_kv_num_per_request;
        status = petps::RpcStatus::kBatchTooLarge;
      } else {
        for (std::size_t i = 0; i < decoded.keys.size(); ++i) {
          cache_ps_->PutSingleParameter(
              decoded.keys[i],
              decoded.values.data() + i * decoded.embedding_dim,
              decoded.embedding_dim,
              thread_id);
        }
      }
    }

    const std::int32_t code = static_cast<std::int32_t>(status);
    auto* ack_buf           = dsm_->get_rdma_buffer();
    std::memcpy(ack_buf, &code, sizeof(code));
    WriteMayflyResponse(
        request,
        ack_buf,
        request.receive_gaddr,
        sizeof(code),
        true,
        petps::WR_ID_PUT,
        thread_id);
  }

  void RpcPsPut(RawMessage* recv, int thread_id) {
    HandlePsPut(RequestViewFromRawMessage(recv), thread_id);
  }

  void HandlePsGet(const RequestView& request, int thread_id) {
    const bool trace_get = ShouldTraceRdmaGet();
    const auto trace_start =
        trace_get ? std::chrono::steady_clock::now()
                  : std::chrono::steady_clock::time_point{};
    const bool perf_condition = (thread_id == 0);
    auto& sourcelist          = sourcelists_[thread_id];

    epoch_manager_->Protect();

    if (perf_condition)
      get_parameter_timer_.start();
    Slice extra_data = request.payload;

    int batch_get_kv_count = extra_data.len / sizeof(uint64_t);
    tp[thread_id][0] += batch_get_kv_count;
    base::ConstArray<uint64_t> keys(
        (uint64_t*)extra_data.s, batch_get_kv_count);
    if (ShouldValidateRouting()) {
      for (auto each : keys) {
        CHECK_EQ(XPostoffice::GetInstance()->ServerID(),
                 ShardManager::KeyPartition(each))
            << each << " not belong to this PS; "
            << "sended from client node_id = " << (int)request.node_id;
      }
    }
#ifdef RPC_DEBUG
    LOG(INFO) << "recv->msg_size=" << extra_data.len;
    LOG(INFO) << "server batch gets: " << keys.Debug();
#endif
    CHECK_LE(batch_get_kv_count, FLAGS_max_kv_num_per_request);
    const auto trace_after_parse =
        trace_get ? std::chrono::steady_clock::now()
                  : std::chrono::steady_clock::time_point{};

    const int embedding_dim = FLAGS_value_size / sizeof(float);
    const std::size_t response_bytes =
        petps::FixedSlotResponseBytes(batch_get_kv_count, FLAGS_value_size);
    auto* buf = dsm_->get_rdma_buffer();

    if (perf_condition)
      index_timer_.start();
    bool flat_get_ok = true;
    if (response_bytes <= FLAGS_rdma_per_thread_response_limit_bytes) {
      flat_get_ok = cache_ps_->GetParameterFlat(
          keys,
          reinterpret_cast<float*>(buf),
          batch_get_kv_count,
          embedding_dim,
          thread_id);
    }
    if (perf_condition)
      index_timer_.end();
    const auto trace_after_index =
        trace_get ? std::chrono::steady_clock::now()
                  : std::chrono::steady_clock::time_point{};

#ifdef RPC_DEBUG
    if (response_bytes <= FLAGS_rdma_per_thread_response_limit_bytes &&
        flat_get_ok) {
      for (int i = 0; i < batch_get_kv_count; i++) {
        float* slot = reinterpret_cast<float*>(buf + i * FLAGS_value_size);
        XDebug::AssertTensorEq(
            slot,
            embedding_dim,
            keys[i],
            base::SFormat("server embedding check error, key is {}", keys[i]));
      }
    }
#endif
    if (perf_condition)
      value_timer_.start();

    if (response_bytes > FLAGS_rdma_per_thread_response_limit_bytes) {
      LOG(ERROR) << "component=rdma_server event=batch_too_large shard="
                 << XPostoffice::GetInstance()->ServerID() << " thread_id="
                 << thread_id << " key_count=" << batch_get_kv_count
                 << " response_bytes=" << response_bytes << " limit_bytes="
                 << FLAGS_rdma_per_thread_response_limit_bytes;
      auto* status_word =
          reinterpret_cast<std::int32_t*>(dsm_->get_rdma_buffer());
      *status_word =
          static_cast<std::int32_t>(petps::RpcStatus::kBatchTooLarge);
      WriteMayflyResponse(
          request,
          reinterpret_cast<const char*>(status_word),
          request.receive_gaddr,
          sizeof(std::int32_t),
          true,
          petps::WR_ID_GET,
          thread_id);
      epoch_manager_->UnProtect();
      return;
    }

    auto* status_word = reinterpret_cast<std::int32_t*>(
        buf + batch_get_kv_count * FLAGS_value_size);
    if (flat_get_ok) {
      *status_word = static_cast<std::int32_t>(petps::RpcStatus::kOk);
    } else {
      *status_word =
          static_cast<std::int32_t>(petps::RpcStatus::kValueSizeMismatch);
    }

    epoch_manager_->UnProtect();
    GlobalAddress gaddr = request.receive_gaddr;
    const auto trace_before_write =
        trace_get ? std::chrono::steady_clock::now()
                  : std::chrono::steady_clock::time_point{};
    WriteMayflyResponse(
        request, buf, gaddr, response_bytes, true, petps::WR_ID_GET, thread_id);
    if (trace_get) {
      const auto trace_done = std::chrono::steady_clock::now();
      auto& stats           = request.descriptor_doorbell
                                ? DescriptorGetServerHandleTraceStats()
                                : RawGetServerHandleTraceStats();
      stats.Add(
          request.descriptor_doorbell ? "descriptor_doorbell" : "raw_message",
          ToNs(trace_after_parse - trace_start),
          ToNs(trace_after_index - trace_after_parse),
          ToNs(trace_before_write - trace_after_index),
          ToNs(trace_done - trace_before_write),
          ToNs(trace_done - trace_start));
    }
    if (perf_condition)
      value_timer_.end();

#ifdef RPC_DEBUG
    LOG(INFO) << "RPC done";
#endif
    if (perf_condition)
      get_parameter_timer_.end();
  }

  void RpcPsGet(RawMessage* recv, int thread_id) {
    HandlePsGet(RequestViewFromRawMessage(recv), thread_id);
  }

  void RpcPsDescriptorDoorbell(const petps::RdmaDescriptorRequest& request,
                               Slice inline_payload,
                               int thread_id) {
    if (request.op ==
        static_cast<std::uint16_t>(petps::RdmaDescriptorOp::kGet)) {
      HandlePsGet(
          RequestView{
              RpcType::GET,
              static_cast<NodeIDType>(request.client_node_id),
              static_cast<ThreadIDType>(request.client_thread_id),
              request.response_gaddr,
              inline_payload,
              true,
          },
          thread_id);
    } else if (request.op ==
               static_cast<std::uint16_t>(petps::RdmaDescriptorOp::kPut)) {
      HandlePsPut(
          RequestView{
              RpcType::PUT,
              static_cast<NodeIDType>(request.client_node_id),
              static_cast<ThreadIDType>(request.client_thread_id),
              request.status_gaddr,
              inline_payload,
              true,
          },
          thread_id);
    } else {
      LOG(ERROR) << "unknown descriptor op=" << request.op;
    }
  }

  bool DecodeDescriptorCompletion(petps::RawVerbsTransport* raw_transport,
                                  const petps::RawVerbsCompletion& completion,
                                  DescriptorTask* task) {
    if (raw_transport == nullptr || task == nullptr || !completion.has_imm) {
      return false;
    }
    const std::uint32_t slot_id = completion.imm_data;
    petps::RdmaDescriptorRequest request{};
    request.client_node_id   = static_cast<std::uint16_t>(completion.wr_id);
    request.slot_id          = slot_id;
    request.descriptor_gaddr = GlobalAddress{
        static_cast<std::uint16_t>(dsm_->getMyNodeID()),
        FLAGS_rdma_put_v2_push_region_offset +
            (static_cast<std::uint64_t>(request.client_node_id) *
                 static_cast<std::uint64_t>(
                     FLAGS_rdma_put_v2_push_slots_per_client) +
             slot_id) *
                FLAGS_rdma_put_v2_push_slot_bytes};
    petps::RdmaDescriptorLaneConfig lane_config{};
    lane_config.region_offset = FLAGS_rdma_put_v2_push_region_offset;
    lane_config.slot_bytes    = FLAGS_rdma_put_v2_push_slot_bytes;
    lane_config.slots_per_client =
        static_cast<std::uint32_t>(FLAGS_rdma_put_v2_push_slots_per_client);
    lane_config.machine_count =
        static_cast<std::uint32_t>(dsm_->get_conf()->machineNR);
    std::string error;
    if (!petps::ValidateDescriptorLane(request, lane_config, &error)) {
      LOG(ERROR) << "invalid descriptor lane: " << error;
      return false;
    }
    const char* payload = static_cast<const char*>(
        raw_transport->LocalPointer(request.descriptor_gaddr));
    if (!petps::DecodeRdmaDescriptorRequest(
            std::string_view(payload, sizeof(petps::RdmaDescriptorRequest)),
            &request,
            &error)) {
      LOG(ERROR) << "descriptor decode error: " << error;
      return false;
    }
    if (!petps::ValidateDescriptorLane(request, lane_config, &error)) {
      LOG(ERROR) << "decoded descriptor lane mismatch: " << error;
      return false;
    }
    if (request.payload_bytes > FLAGS_rdma_put_v2_push_slot_bytes -
                                    sizeof(petps::RdmaDescriptorRequest)) {
      LOG(ERROR) << "descriptor inline payload too large request_id="
                 << request.request_id
                 << " payload_bytes=" << request.payload_bytes
                 << " slot_bytes=" << FLAGS_rdma_put_v2_push_slot_bytes;
      return false;
    }
    task->request              = request;
    task->inline_payload       = payload + sizeof(petps::RdmaDescriptorRequest);
    task->inline_payload_bytes = request.payload_bytes;
    return true;
  }

  void DescriptorPollLoop(int thread_id) {
    auto* raw_transport =
        raw_transports_[static_cast<std::size_t>(thread_id)].get();
    CHECK(raw_transport != nullptr)
        << "missing descriptor raw transport for thread_id=" << thread_id;
    while (1) {
      const bool trace_get = ShouldTraceRdmaGet();
      const auto trace_start =
          trace_get ? std::chrono::steady_clock::now()
                    : std::chrono::steady_clock::time_point{};
      petps::RawVerbsCompletion completion{};
      if (!raw_transport->Poll(&completion, FLAGS_rdma_wait_timeout_ms)) {
        std::this_thread::yield();
        continue;
      }
      const auto trace_after_poll =
          trace_get ? std::chrono::steady_clock::now()
                    : std::chrono::steady_clock::time_point{};
      DescriptorTask task;
      if (!DecodeDescriptorCompletion(raw_transport, completion, &task)) {
        continue;
      }
      const auto trace_after_decode =
          trace_get ? std::chrono::steady_clock::now()
                    : std::chrono::steady_clock::time_point{};
      RpcPsDescriptorDoorbell(
          task.request,
          Slice(task.inline_payload, task.inline_payload_bytes),
          thread_id);
      if (trace_get &&
          task.request.op ==
              static_cast<std::uint16_t>(petps::RdmaDescriptorOp::kGet)) {
        const auto trace_done = std::chrono::steady_clock::now();
        DescriptorGetServerEnvelopeTraceStats().Add(
            ToNs(trace_after_poll - trace_start),
            ToNs(trace_after_decode - trace_after_poll),
            ToNs(trace_done - trace_after_decode),
            ToNs(trace_done - trace_start));
      }
    }
  }

  void PollingThread(int thread_id) {
    auto_bind_core(0);
    dsm_->registerThread();
    VLOG(1) << "component=rdma_server event=polling_thread_ready thread_id="
            << thread_id;
    const int ready_threads = registered_polling_threads_.fetch_add(1) + 1;
    if (transport_mode_ != petps::RdmaTransportMode::kDescriptorDoorbell &&
        ready_threads == thread_count_) {
      PublishReadyKeys();
    }
    if (transport_mode_ == petps::RdmaTransportMode::kDescriptorDoorbell) {
      if (ready_threads == thread_count_) {
        PublishDescriptorWorkerThreads();
      }
      auto* raw_transport =
          raw_transports_[static_cast<std::size_t>(thread_id)].get();
      if (raw_transport == nullptr) {
        const int serving_thread_count = static_cast<int>(
            petps::GetRdmaDescriptorServingThreadIDs(thread_count_).size());
        while (!petps::CanPublishRdmaDescriptorReady(
            registered_polling_threads_.load(std::memory_order_acquire),
            thread_count_,
            raw_connected_endpoints_.load(std::memory_order_acquire) ==
                serving_thread_count)) {
          std::this_thread::yield();
        }
        PublishReadyKeys();
        while (1) {
          std::this_thread::sleep_for(std::chrono::seconds(1));
        }
      }
      raw_transport->PublishAndConnect();
      raw_connected_endpoints_.fetch_add(1, std::memory_order_acq_rel);
      const int serving_thread_count = static_cast<int>(
          petps::GetRdmaDescriptorServingThreadIDs(thread_count_).size());
      while (!petps::CanPublishRdmaDescriptorReady(
          registered_polling_threads_.load(std::memory_order_acquire),
          thread_count_,
          raw_connected_endpoints_.load(std::memory_order_acquire) ==
              serving_thread_count)) {
        std::this_thread::yield();
      }
      PublishReadyKeys();
      while (!ready_published_.load(std::memory_order_acquire)) {
        std::this_thread::yield();
      }
      DescriptorPollLoop(thread_id);
    }
    auto msg = RawMessage::get_new_msg();

    while (1) {
      msg->clear();
      uint64_t wr_id;
      RawMessage* recv;
      do {
        recv = dsm_->rpc_fast_wait(&wr_id);
        if (recv == nullptr && wr_id == petps::WR_ID_SG_GET) {
          // RECSTORE_LOG_EVERY_MS(ERROR, 1000)
          //     << "MaxPendingEpochNumPerThread = "
          //     << epoch_manager_->MaxPendingEpochNumPerThread();
          epoch_manager_->UnProtect();
        }
      } while (nullptr == recv);

      if (recv->type == GET_SERVER_THREADIDS) {
        RpcGetServerServingThreadIDs(recv);
      } else if (recv->type == PUT) {
        RpcPsPut(recv, thread_id);
      } else if (recv->type == GET) {
        RpcPsGet(recv, thread_id);
      } else {
        LOG(FATAL) << "unknown message type";
      }
    }
  }

private:
  std::vector<std::vector<SourceList>> sourcelists_;
  CachePS* cache_ps_;
  std::vector<std::thread> threads_;
  int thread_count_;
  DSM* dsm_;
  std::atomic_int registered_polling_threads_{0};
  std::atomic<bool> ready_published_{false};
  xmh::Timer get_parameter_timer_;
  xmh::Timer index_timer_;
  xmh::Timer value_timer_;

  base::epoch::EpochManager* epoch_manager_;
  petps::RdmaTransportMode transport_mode_ =
      petps::RdmaTransportMode::kRawMessage;
  std::vector<std::unique_ptr<petps::RawVerbsTransport>> raw_transports_;
  std::atomic<int> raw_connected_endpoints_{0};

  constexpr static int kMaxThread = 128;
  uint64_t tp[kMaxThread][8];
  std::uint64_t descriptor_dsm_write_counters_[kMaxThread] = {};
};
} // namespace recstore

int main(int argc, char* argv[]) {
  folly::init(&argc, &argv);
  if (ShouldTraceRdmaGet()) {
    std::cerr << "component=rdma_get_trace side=server event=enabled interval="
              << RdmaGetTraceInterval() << std::endl;
  }
  xmh::Reporter::StartReportThread();

  base::PMMmapRegisterCenter::GetConfig().use_dram = FLAGS_use_dram;
  base::PMMmapRegisterCenter::GetConfig().numa_id  = FLAGS_numa_id;

  extern int global_socket_id;
  global_socket_id = FLAGS_numa_id;
  LOG(INFO) << "set NUMA ID = " << FLAGS_numa_id;

  const std::string config_path =
      FLAGS_config_path.empty()
          ? base::ResolveRecStoreConfigPath().string()
          : FLAGS_config_path;
  std::ifstream config_file(config_path);
  if (!config_file.is_open()) {
    LOG(FATAL) << "Cannot open config file: " << config_path;
  }
  nlohmann::json config;
  config_file >> config;

  auto cache_ps = std::make_unique<CachePS>(config["cache_ps"]);

  if (FLAGS_preload) {
    LOG(INFO) << "Loading fake data for preload";
    int64_t ps_capacity = config["cache_ps"]["base_kv_config"]["capacity"];
    cache_ps->LoadFakeData(ps_capacity * FLAGS_warmup_ratio, FLAGS_value_size);
  }

  recstore::PetPSServer parameterServiceImpl(cache_ps.get(), FLAGS_thread_num);
  parameterServiceImpl.Run();

  while (1) {
    auto micro_second1 = base::GetTimestamp();
    uint64_t tp_sum1   = parameterServiceImpl.GetThroughputCounterSum();
    std::this_thread::sleep_for(std::chrono::seconds(1));
    auto micro_second2 = base::GetTimestamp();
    uint64_t tp_sum2   = parameterServiceImpl.GetThroughputCounterSum();
    double tps = (tp_sum2 - tp_sum1) * 1.0 / (micro_second2 - micro_second1);
    printf("throughput %.4f Mkv/s\n", tps);
  }
  return 0;
}
