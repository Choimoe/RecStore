#pragma once
#include <string>
#include <tuple>

#include "base/array.h"
#include "base/json.h"
#include "base/log.h"

namespace recstore {

enum class PSCommand { CLEAR_PS, RELOAD_PS, LOAD_FAKE_DATA };

class BasePSClient {
  json json_config_;

 public:
  explicit BasePSClient(json config) : json_config_(config) {}
  virtual ~BasePSClient() {}

  virtual int GetParameter(const base::ConstArray<uint64_t> &keys,
                           float *values) = 0;

  virtual int AsyncGetParameter(const base::ConstArray<uint64_t> &keys,
                                float *values) = 0;

  virtual int PutParameter(const base::ConstArray<uint64_t> &keys,
                           const std::vector<std::vector<float>> &values) = 0;

  virtual void Command(PSCommand command) = 0;

  virtual uint64_t PrefetchParameter(const base::ConstArray<uint64_t>& keys) = 0;
  virtual bool IsPrefetchDone(uint64_t prefetch_id) = 0;
  virtual void WaitForPrefetch(uint64_t prefetch_id) = 0;
  virtual bool GetPrefetchResult(uint64_t prefetch_id, std::vector<std::vector<float>>* values) = 0;
};

}  // namespace recstore
