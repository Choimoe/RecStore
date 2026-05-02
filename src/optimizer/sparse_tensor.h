#include <string>
#include <vector>
#include "../storage/kv_engine/base_kv.h"

#define TAG_TYPE uint8_t

enum TensorType { PARAMETER = 0, MOMENT_1 = 1, MOMENT_2 = 2 };

class SparseTensor {
private:
  std::string name;
  TensorType type;
  TAG_TYPE tag;
  std::vector<uint64_t> shape;
  BaseKV* kv;
  uint64_t concatKeyAndTag(uint64_t key, TAG_TYPE tag);

public:
  SparseTensor() = default;
  void init(std::string& name,
            TensorType type,
            TAG_TYPE tag,
            std::vector<uint64_t>& shape,
            BaseKV* kv);
  void Get(const uint64_t key, std::string& value, unsigned tid);
  void Put(const uint64_t key, const std::string_view& value, unsigned tid);
  void BatchGet(const std::vector<uint64_t>& keys,
                std::vector<base::ConstArray<float>>* values,
                unsigned tid);
  bool ApplySgdUpdateFlat(
      const base::ConstArray<uint64_t>& keys,
      const float* grads,
      int64_t num_rows,
      int64_t embedding_dim,
      float learning_rate,
      unsigned tid);
};
