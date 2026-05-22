#pragma once
#include <atomic>
#include <cstdint>
#include <memory>
#include <mutex>
#include <string>
#include <thread>
#include <unordered_map>
#include <utility>
#include <vector>

#include "base/array.h"
#include "base/factory.h"
#include "base/log.h"
#include "base_client.h"
#include "raw_verbs_transport.h"
#include "rdma_protocol.h"
#include "rdma_status.h"
#include "rdma_transport_mode.h"
#include "third_party/Mayfly-main/include/DSM.h"

namespace petps {
class PetPSClient : public BaseParameterClient {
public:
  explicit PetPSClient(const std::string& host, int port, int shard)
      : BaseParameterClient(host, port, shard) {
    Init();
  }

  ~PetPSClient() {}

  void Barrier(const std::string& ss, int k) override { dsm_->barrier(ss, k); }

  void InitThread() override {
    LOG(INFO) << "dsm_->registerThread()";
    dsm_->registerThread();
    if (transport_mode_ == RdmaTransportMode::kDescriptorDoorbell) {
      SetCurrentThreadServerThreadIDs(GetServerThreadIDs());
      InitDescriptorTransports();
      std::call_once(server_ready_once_, [this]() { WaitForServerReady(); });
    } else {
      std::call_once(server_ready_once_, [this]() { WaitForServerReady(); });
      SetCurrentThreadServerThreadIDs(GetRawServerThreadIDsForCurrentThread());
    }
  }

  int GetParameter(base::ConstArray<uint64_t> keys,
                   std::vector<std::vector<float>>* values) override;

  // this interface assume all keys with the same embedding dimension
  int GetParameter(base::ConstArray<uint64_t> keys,
                   float* values,
                   bool isAsync,
                   int async_req_id = 0) override;

  std::size_t ResponseBufferBytes(std::size_t key_count) const;

  void* GetReceiveBuffer(size_t size) override;
  void* GetSendBuffer(size_t size);

  inline int shard() const { return shard_; }

  bool QueryRPCFinished(int rpc_id) override;

  void WaitRPCFinish(int rpc_id) override;

  void RevokeRPCResource(int rpc_id) override;

  int PutParameter(const std::vector<uint64_t>& keys,
                   const std::vector<std::vector<float>>& values) override;

  int FakePutParameter(base::ConstArray<uint64_t> keys, float* values) override;

private:
  void WaitForServerReady();
  std::vector<int> GetServerThreadIDs();
  std::vector<int> GetRawServerThreadIDsForCurrentThread();
  void SetCurrentThreadServerThreadIDs(std::vector<int> thread_ids);
  std::vector<int> GetCurrentThreadServerThreadIDs() const;
  int SelectServerThreadID() const;
  std::atomic<int32_t>* GetPollSlot(uint64_t rpc_id) const;
  void Init();
  void InitDescriptorTransports();
  void InitDescriptorSlots();
  RawVerbsTransport* DescriptorTransportForThread(int thread_id) const;
  std::mutex* DescriptorTransportMutexForThread(int thread_id) const;
  int GetParameterDescriptor(
      base::ConstArray<uint64_t> keys, float* values, bool isAsync);
  int PutParameterDescriptor(const std::vector<uint64_t>& keys,
                             const std::vector<std::vector<float>>& values);
  struct DescriptorPendingRpc {
    std::unique_lock<std::mutex> descriptor_slot_guard;
  };
  DSM* dsm_;

  mutable std::mutex rpc_mu_;
  mutable std::mutex get_mu_;
  mutable std::mutex put_mu_;
  mutable std::mutex routing_mu_;
  std::once_flag server_ready_once_;
  std::once_flag raw_server_thread_ids_once_;
  std::atomic<uint64_t> rpcIDAcc_{0};
  mutable std::atomic<uint32_t> round_robin_{0};

  std::vector<int> raw_server_thread_ids_;
  std::unordered_map<std::thread::id, std::vector<int>>
      server_thread_ids_by_thread_;
  std::unordered_map<uint64_t, std::atomic<int32_t>*> rpcId2PollMap_;
  RdmaTransportMode transport_mode_ = RdmaTransportMode::kRawMessage;
  std::shared_ptr<RawVerbsTransport> raw_transport_;
  std::shared_ptr<std::mutex> raw_transport_mu_;
  std::unordered_map<int, std::shared_ptr<RawVerbsTransport>>
      descriptor_transports_by_thread_;
  std::unordered_map<int, std::shared_ptr<std::mutex>>
      descriptor_transport_mus_by_thread_;
  std::atomic<std::uint64_t> descriptor_request_id_{1};
  std::atomic<std::uint64_t> descriptor_slot_cursor_{0};
  RdmaDescriptorDoorbellPostState descriptor_doorbell_post_state_;
  std::vector<char*> descriptor_send_slots_;
  std::vector<std::atomic<std::int32_t>*> descriptor_ack_slots_;
  std::vector<std::unique_ptr<std::mutex>> descriptor_slot_locks_;
  std::unordered_map<uint64_t, DescriptorPendingRpc> descriptor_pending_rpcs_;
};

FACTORY_REGISTER(BaseParameterClient,
                 PetPSClient,
                 PetPSClient,
                 const std::string&,
                 int,
                 int);
} // namespace petps
