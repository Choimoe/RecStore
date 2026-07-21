#pragma once

#include <cstdint>
#include <future>
#include <mutex>
#include <string>
#include <unordered_map>
#include <vector>

#include "base/array.h"
#include "base/flatc.h"
#include "base/init.h"
#include "base/json.h"
#include "ps/base/base_client.h"
#include "ps/base/parameters.h"
#include "ps.grpc.pb.h"
#include "ps.pb.h"
#include "base/tensor.h"

using grpc::Channel;
using grpc::ClientContext;
using grpc::Status;
using recstoreps::CommandRequest;
using recstoreps::CommandResponse;
using recstoreps::GetParameterRequest;
using recstoreps::GetParameterResponse;
using recstoreps::PSCommand;
using recstoreps::PutParameterRequest;
using recstoreps::PutParameterResponse;

using base::ConstArray;
using json = nlohmann::json;

static const int MAX_PARAMETER_BATCH = 2000;

struct PrefetchBatch {
  PrefetchBatch(int request_num) {
    batch_size_ = request_num;
    key_sizes_.resize(request_num);
    status_.resize(request_num);
    contexts_.resize(request_num);
    requests_.resize(request_num);
    responses_.resize(request_num);
    response_readers_.resize(request_num);
    cqs_             = std::make_unique<grpc::CompletionQueue>();
    completed_count_ = 0;
  }

  PrefetchBatch(PrefetchBatch&& other) noexcept
      : key_sizes_(std::move(other.key_sizes_)),
        status_(std::move(other.status_)),
        contexts_(std::move(other.contexts_)),
        requests_(std::move(other.requests_)),
        responses_(std::move(other.responses_)),
        response_readers_(std::move(other.response_readers_)),
        batch_size_(other.batch_size_),
        cqs_(std::move(other.cqs_)),
        completed_count_(other.completed_count_) {
    other.batch_size_ = 0;
  }
  PrefetchBatch(const PrefetchBatch&)            = delete;
  PrefetchBatch& operator=(const PrefetchBatch&) = delete;

  std::vector<int> key_sizes_;
  std::vector<Status> status_;
  std::vector<std::unique_ptr<ClientContext>> contexts_;
  std::vector<GetParameterRequest> requests_;
  std::vector<GetParameterResponse> responses_;
  std::vector<
      std::unique_ptr<grpc::ClientAsyncResponseReader<GetParameterResponse>>>
      response_readers_;

  int batch_size_;
  int completed_count_;
  std::unique_ptr<grpc::CompletionQueue> cqs_;
};

struct PrewriteBatch {
  PrewriteBatch(int request_num) {
    batch_size_ = request_num;
    key_sizes_.resize(request_num);
    status_.resize(request_num);
    contexts_.resize(request_num);
    requests_.resize(request_num);
    responses_.resize(request_num);
    response_readers_.resize(request_num);
    cqs_             = std::make_unique<grpc::CompletionQueue>();
    completed_count_ = 0;
  }

  PrewriteBatch(PrewriteBatch&& other) noexcept
      : key_sizes_(std::move(other.key_sizes_)),
        status_(std::move(other.status_)),
        contexts_(std::move(other.contexts_)),
        requests_(std::move(other.requests_)),
        responses_(std::move(other.responses_)),
        response_readers_(std::move(other.response_readers_)),
        batch_size_(other.batch_size_),
        cqs_(std::move(other.cqs_)),
        completed_count_(other.completed_count_) {
    other.batch_size_ = 0;
  }
  PrewriteBatch(const PrewriteBatch&)            = delete;
  PrewriteBatch& operator=(const PrewriteBatch&) = delete;

  std::vector<int> key_sizes_;
  std::vector<Status> status_;
  std::vector<std::unique_ptr<ClientContext>> contexts_;
  std::vector<PutParameterRequest> requests_;
  std::vector<PutParameterResponse> responses_;
  std::vector<
      std::unique_ptr<grpc::ClientAsyncResponseReader<PutParameterResponse>>>
      response_readers_;

  int batch_size_;
  int completed_count_;
  std::unique_ptr<grpc::CompletionQueue> cqs_;
};

class GRPCParameterClient : public recstore::BasePSClient {
public:
  // New constructor with JSON config
  explicit GRPCParameterClient(json config);

  // Legacy constructor for backward compatibility
  explicit GRPCParameterClient(const std::string& host, int port, int shard);

  ~GRPCParameterClient() {}

  // BasePSClient pure virtual implementations
  virtual int
  GetParameter(const base::ConstArray<uint64_t>& keys, float* values) override;

  int AsyncGetParameter(const base::ConstArray<uint64_t>& keys,
                        float* values) override;

  int PutParameter(const base::ConstArray<uint64_t>& keys,
                   const std::vector<std::vector<float>>& values) override;

  void Command(recstore::PSCommand command) override;

  // Legacy API methods
  int GetParameter(const base::ConstArray<uint64_t>& keys,
                   std::vector<std::vector<float>>* values);
  bool GetParameter(const base::ConstArray<unsigned int>& keys,
                    std::vector<std::vector<float>>* values);

  inline int shard() const { return shard_; }

  bool ClearPS();

  bool LoadFakeData(int64_t data);

  // Write n bytes of random floats to storage at key 0. n must be a positive
  // multiple of sizeof(float).
  bool DumpFakeData(int64_t n);

  bool LoadCkpt(const std::vector<std::string>& model_config_path,
                const std::vector<std::string>& emb_file_path);

  bool
  SaveCheckpoint(const std::string& path, const std::string& metadata) override;
  bool
  LoadCheckpoint(const std::string& path, const std::string& metadata) override;

  bool PutParameter(const std::vector<uint64_t>& keys,
                    const std::vector<std::vector<float>>& values);

  int UpdateParameter(const std::string& table_name,
                      const base::ConstArray<uint64_t>& keys,
                      const std::vector<std::vector<float>>* grads);
  int UpdateParameterFlat(const std::string& table_name,
                          const base::ConstArray<uint64_t>& keys,
                          const float* grads,
                          int64_t num_rows,
                          int64_t embedding_dim) override;

  int InitEmbeddingTable(const std::string& table_name,
                         const recstore::EmbeddingTableConfig& config);

  uint64_t PrefetchParameter(const base::ConstArray<uint64_t>& keys);
  bool IsPrefetchDone(uint64_t prefetch_id);
  void WaitForPrefetch(uint64_t prefetch_id);
  bool GetPrefetchResult(uint64_t prefetch_id,
                         std::vector<std::vector<float>>* values);
  bool GetPrefetchResultFlat(uint64_t prefetch_id,
                             std::vector<float>* values,
                             int64_t* num_rows,
                             int64_t embedding_dim) override;
  // Embeddings are vectors here; Get(float*) uses a flat buffer for legacy
  // callers
  virtual uint64_t EmbWriteAsync(const base::ConstArray<uint64_t>& keys,
                                 const std::vector<std::vector<float>>& values);
  virtual bool IsWriteDone(uint64_t write_id);
  virtual void WaitForWrite(uint64_t write_id);

protected:
  bool Initialize() { return true; }
  std::string host_;
  int port_;
  int shard_;
  int nr_clients_;
  std::vector<float> cache_;
  std::vector<int32_t> offset_;
  std::vector<int> get_param_key_sizes_;
  std::vector<Status> get_param_status_;
  std::vector<GetParameterRequest> get_param_requests_;
  std::vector<GetParameterResponse> get_param_responses_;
  std::vector<std::unique_ptr<grpc::ClientContext>> get_param_contexts_;
  std::vector<
      std::unique_ptr<grpc::ClientAsyncResponseReader<GetParameterResponse>>>
      get_param_resonse_readers_;
  std::shared_ptr<Channel> channel_;
  std::vector<std::unique_ptr<recstoreps::ParameterService::Stub>> stubs_;
  std::unique_ptr<grpc::CompletionQueue> cq_;

private:
  std::mutex prefetch_mu_;
  std::unordered_map<uint64_t, struct PrefetchBatch> prefetch_batches_;
  std::unordered_map<uint64_t, struct PrewriteBatch> prewrite_batches_;
  // start from 1
  uint64_t next_prefetch_id_ = 1;
  uint64_t next_prewrite_id_ = 1;
};
