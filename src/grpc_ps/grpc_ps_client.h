#pragma once

#include <folly/Portability.h>
#include <folly/executors/CPUThreadPoolExecutor.h>
#include <folly/init/Init.h>

#include <cstdint>
#include <future>
#include <string>
#include <unordered_map>
#include <vector>

#include "base/array.h"
#include "base/flatc.h"
#include "base/json.h"
#include "base_ps/base_client.h"
#include "base_ps/parameters.h"
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
        requests_.resize(request_num);
        responses_.resize(request_num);
        response_readers_.resize(request_num);
        cqs_ = std::make_unique<grpc::CompletionQueue>();
        completed_count_ = 0;
    }

    PrefetchBatch(PrefetchBatch&& other) noexcept
        : key_sizes_(std::move(other.key_sizes_)),
          status_(std::move(other.status_)),
          requests_(std::move(other.requests_)),
          responses_(std::move(other.responses_)),
          response_readers_(std::move(other.response_readers_)),
          batch_size_(other.batch_size_),
          cqs_(std::move(other.cqs_)),
          completed_count_(other.completed_count_) 
    {
        other.batch_size_ = 0;
    }
    PrefetchBatch(const PrefetchBatch&) = delete;
    PrefetchBatch& operator=(const PrefetchBatch&) = delete;

    std::vector<int> key_sizes_;
    std::vector<Status> status_;
    std::vector<GetParameterRequest> requests_;
    std::vector<GetParameterResponse> responses_;
    std::vector<
      std::unique_ptr<grpc::ClientAsyncResponseReader<GetParameterResponse>>>
      response_readers_;

    int batch_size_;
    int completed_count_;
    std::unique_ptr<grpc::CompletionQueue> cqs_;
};

class GRPCParameterClient : public recstore::BasePSClient {
public:
  // 新的构造函数，接收 json 配置参数
  explicit GRPCParameterClient(json config);

  // 保留原有的构造函数以保持向后兼容
  explicit GRPCParameterClient(const std::string& host, int port, int shard);

  ~GRPCParameterClient() {}

  // 实现 BasePSClient 的纯虚函数
  virtual int GetParameter(const base::ConstArray<uint64_t>& keys, float* values) override;

  int AsyncGetParameter(const base::ConstArray<uint64_t>& keys, float* values) override;

  int PutParameter(const base::ConstArray<uint64_t>& keys, const std::vector<std::vector<float>>& values) override;

  void Command(recstore::PSCommand command) override;

  // 原有的接口方法
  int GetParameter(const base::ConstArray<uint64_t>& keys, std::vector<std::vector<float>>* values);
  bool GetParameter(const base::ConstArray<unsigned int>& keys, std::vector<std::vector<float>>* values);

  inline int shard() const { return shard_; }

  bool ClearPS();

  bool LoadFakeData(int64_t data);

  bool LoadCkpt(const std::vector<std::string> &model_config_path,
                const std::vector<std::string> &emb_file_path);

  bool PutParameter(const std::vector<uint64_t> &keys,
                    const std::vector<std::vector<float>> &values);


  uint64_t PrefetchParameter(const base::ConstArray<uint64_t>& keys);
  bool IsPrefetchDone(uint64_t prefetch_id);
  void WaitForPrefetch(uint64_t prefetch_id);
  bool GetPrefetchResult(uint64_t prefetch_id, std::vector<std::vector<float>>* values);

  virtual uint64_t EmbWriteAsync(const base::RecTensor& keys, const base::RecTensor& values);
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
  std::vector<
      std::unique_ptr<grpc::ClientAsyncResponseReader<GetParameterResponse>>>
      get_param_resonse_readers_;
  std::shared_ptr<Channel> channel_;
  std::vector<std::unique_ptr<recstoreps::ParameterService::Stub>> stubs_;
  grpc::CompletionQueue cq;

 private:
    
    std::unordered_map<uint64_t, struct PrefetchBatch> prefetch_batches_; 
    // start from 1
    uint64_t next_prefetch_id_ = 1; 
};
