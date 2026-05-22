#pragma once

#include <memory>
#include <mutex>
#include <shared_mutex>
#include <string>
#include <thread>
#include <unordered_map>
#include <vector>

#include "base/json.h"
#include "ps/base/base_client.h"
#include "ps/rdma/allshards_ps_client.h"
#include "ps/rdma/petps_client.h"

namespace recstore {

void InitializeRdmaProcessRuntime();

class RDMAPSClientAdapter : public BasePSClient {
public:
  explicit RDMAPSClientAdapter(json config);
  ~RDMAPSClientAdapter() override = default;

  int GetParameter(const base::ConstArray<uint64_t>& keys,
                   float* values) override;
  int PutParameter(const base::ConstArray<uint64_t>& keys,
                   const std::vector<std::vector<float>>& values) override;
  int UpdateParameter(const std::string& table_name,
                      const base::ConstArray<uint64_t>& keys,
                      const std::vector<std::vector<float>>* grads) override;
  int UpdateParameterFlat(const std::string& table_name,
                          const base::ConstArray<uint64_t>& keys,
                          const float* grads,
                          int64_t num_rows,
                          int64_t embedding_dim) override;
  int InitEmbeddingTable(const std::string& table_name,
                         const EmbeddingTableConfig& config) override;
  int AsyncGetParameter(const base::ConstArray<uint64_t>& keys,
                        float* values) override;
  void Command(PSCommand command) override;
  uint64_t PrefetchParameter(const base::ConstArray<uint64_t>& keys) override;
  bool IsPrefetchDone(uint64_t prefetch_id) override;
  void WaitForPrefetch(uint64_t prefetch_id) override;
  bool GetPrefetchResult(uint64_t prefetch_id,
                         std::vector<std::vector<float>>* values) override;
  bool GetPrefetchResultFlat(uint64_t prefetch_id,
                             std::vector<float>* values,
                             int64_t* num_rows,
                             int64_t embedding_dim) override;

private:
  struct TableState {
    EmbeddingTableConfig config;
  };

  struct PrefetchChunk {
    float* buffer             = nullptr;
    int rpc_id                = -1;
    int64_t key_count         = 0;
    std::size_t output_offset = 0;
  };

  struct PrefetchState {
    float* buffer         = nullptr;
    int rpc_id            = -1;
    int64_t key_count     = 0;
    int64_t embedding_dim = 0;
    std::vector<PrefetchChunk> chunks;
  };

  void EnsureClientInitialized();
  void EnsureThreadInitialized();
  void EnsureTableReady(const std::string& table_name, int64_t embedding_dim);
  int64_t DefaultEmbeddingDimOrThrow() const;
  PrefetchState GetPrefetchState(uint64_t prefetch_id);
  void MarkPrefetchConsumed(uint64_t prefetch_id);

  json config_;
  std::mutex init_mu_;
  std::mutex thread_init_mu_;
  std::mutex operation_mu_;
  std::shared_mutex runtime_mu_;
  std::mutex state_mu_;
  bool initialized_ = false;
  std::vector<std::unique_ptr<petps::PetPSClient>> shard_clients_;
  std::unique_ptr<AllShardsParameterClientWrapper> multi_client_;
  BaseParameterClient* client_ = nullptr;
  std::unordered_map<std::string, TableState> tables_;
  std::unordered_map<uint64_t, PrefetchState> prefetches_;
  uint64_t next_prefetch_id_ = 1;
  static thread_local float* sync_get_buffer_;
  static thread_local std::size_t sync_get_buffer_bytes_;
};

} // namespace recstore
