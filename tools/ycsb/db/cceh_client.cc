// cceh_db.cc
#include <atomic>
#include <cassert>
#include <cstring>
#include <mutex>
#include <string>
#include <string_view>
#include <sys/stat.h>
#if defined(_MSC_VER)
#  include "direct.h"
#  define mkdir(x, y) _mkdir(x)
#endif

#include "core/db.h"
#include "core/core_workload.h"
#include "core/db_factory.h"
#include "utils/properties.h"
#include "utils/utils.h"

// RecStore headers
#include "../../src/storage/kv_engine/base_kv.h"
#include "../../src/base/factory.h"
#include "../../src/memory/shm_file.h"
#include "../../src/storage/kv_engine/engine_cceh.h"

using ycsbc::DB;

namespace ycsbc {

static std::atomic<unsigned> g_next_tid{0};
static std::atomic<unsigned> g_tid_limit{0};
static inline unsigned ThreadTid() {
  thread_local unsigned t = g_next_tid.fetch_add(1, std::memory_order_relaxed);
  unsigned lim = g_tid_limit.load(std::memory_order_relaxed);
  return lim ? (t % lim) : t;
}

// FNV-1a 64
static inline uint64_t fnv1a64(const std::string& s) {
  const uint64_t O = 1469598103934665603ull, P = 1099511628211ull;
  uint64_t h = O;
  for (unsigned char c : s) {
    h ^= c;
    h *= P;
  }
  return h;
}

// 兼容 "user12345"；若非纯数字落到 FNV
static inline uint64_t ToKey(const std::string& k) {
  try {
    size_t pos = 0;
    uint64_t v = std::stoull(k, &pos, 10);
    if (pos == k.size())
      return v;
  } catch (...) {
  }
  return fnv1a64(k);
}

constexpr const char* EMBEDDING_FIELD_NAME = "embedding";

class CCEHDB : public DB {
public:
  CCEHDB() = default;
  ~CCEHDB() override { Cleanup(); }

  void Init() override {
    const utils::Properties& p = *props_;
    std::lock_guard<std::mutex> lg(mu_);

    if (ref_cnt_++ > 0)
      return;

    field_cnt_ =
        std::stoi(p.GetProperty(ycsbc::CoreWorkload::FIELD_COUNT_PROPERTY,
                                ycsbc::CoreWorkload::FIELD_COUNT_DEFAULT));

    base::PMMmapRegisterCenter::GetConfig().use_dram = true; // 简化部署

    std::string path = p.GetProperty(
        "cceh.path",
        "tools/ycsb/data-store");
    size_t capacity = std::stoull(p.GetProperty("cceh.capacity", "16777216"));
    value_size_     = std::stoull(p.GetProperty("cceh.value_size", "1000"));
    const std::string mode = p.GetProperty("cceh.mode", "compat");
    embedding_mode_ = (mode == "embedding");
    unsigned thread_count = 1;
    try {
      thread_count = std::max(
          1u, (unsigned)std::stoul(p.GetProperty("cceh.threadcount", "1")));
    } catch (...) {
      thread_count = 1;
    }
    g_tid_limit.store(thread_count, std::memory_order_relaxed);

    if (mkdir(path.c_str(), 0775) && errno != EEXIST) {
      throw utils::Exception(std::string("mkdir failed: ") + strerror(errno));
    }

    BaseKVConfig cfg;
    cfg.num_threads_              = thread_count;
    cfg.json_config_["path"]       = path;
    cfg.json_config_["capacity"]   = capacity;
    cfg.json_config_["value_size"] = value_size_;
    cfg.json_config_["queue_cnt"] =
        std::stoi(p.GetProperty("cceh.queue_cnt", "512"));
    cfg.json_config_["io_backend_type"] =
        p.GetProperty("cceh.io_backend_type", "IOURING");

    engine_ = new KVEngineCCEH(cfg);
  }

  void Cleanup() override {
    std::lock_guard<std::mutex> lg(mu_);
    if (ref_cnt_ == 0)
      return;
    if (--ref_cnt_ == 0) {
      delete engine_;
      engine_ = nullptr;
      g_next_tid.store(0, std::memory_order_relaxed);
      g_tid_limit.store(0, std::memory_order_relaxed);
    }
  }

  Status Read(const std::string&,
              const std::string& key,
              const std::vector<std::string>* fields,
              std::vector<Field>& result) override {
    if (!engine_)
      return kError;
    std::string blob;
    engine_->Get(ToKey(key), blob, ThreadTid());
    if (blob.empty())
      return kNotFound;
    if (embedding_mode_) {
      ReturnEmbedding(result, blob);
      return kOK;
    }
    if (fields) {
      DeserializeRowFilter(&result, blob.data(), blob.size(), *fields);
    } else {
      DeserializeRow(&result, blob.data(), blob.size());
    }
    return kOK;
  }

  Status BatchRead(const std::string&,
                   const std::vector<std::string>& keys,
                   const std::vector<std::string>* fields,
                   std::vector<std::vector<Field>>& result) override {
    if (!engine_)
      return kError;
    if (fields != nullptr)
      return DB::BatchRead("", keys, fields, result);
    std::vector<uint64_t> numeric_keys;
    numeric_keys.reserve(keys.size());
    for (const auto& key : keys)
      numeric_keys.push_back(ToKey(key));
    std::vector<base::ConstArray<float>> values;
    engine_->BatchGet(base::ConstArray<uint64_t>(numeric_keys), &values, ThreadTid());
    if (embedding_mode_) {
      if (values.size() != keys.size())
        return kError;
      result.clear();
      result.reserve(values.size());
      for (const auto& value : values) {
        if (value.Size() == 0)
          return kNotFound;
        result.emplace_back();
        result.back().push_back(
            {EMBEDDING_FIELD_NAME,
             std::string(reinterpret_cast<const char*>(value.Data()),
                         value.Size() * sizeof(float))});
      }
    }
    return values.size() == keys.size() ? kOK : kError;
  }

  Status Scan(const std::string&,
              const std::string&,
              int,
              const std::vector<std::string>*,
              std::vector<std::vector<Field>>&) override {
    return kNotFound;
  }

  Status Update(const std::string&,
                const std::string& key,
                std::vector<Field>& values) override {
    if (!engine_)
      return kError;
    if (embedding_mode_) {
      std::string out;
      SerializeEmbedding(values, out);
      engine_->Put(
          ToKey(key), std::string_view(out.data(), out.size()), ThreadTid());
      return kOK;
    }
    std::string blob;
    engine_->Get(ToKey(key), blob, ThreadTid());
    if (blob.empty())
      return kNotFound;

    std::vector<Field> cur;
    DeserializeRow(&cur, blob.data(), blob.size());
    for (auto& nf : values) {
      bool found = false;
      for (auto& cf : cur)
        if (cf.name == nf.name) {
          cf.value = nf.value;
          found    = true;
          break;
        }
      assert(found);
    }
    std::string out;
    SerializeRow(cur, &out);
    engine_->Put(
        ToKey(key), std::string_view(out.data(), out.size()), ThreadTid());
    return kOK;
  }

  Status Insert(const std::string&,
                const std::string& key,
                std::vector<Field>& values) override {
    if (!engine_)
      return kError;
    std::string out;
    if (embedding_mode_)
      SerializeEmbedding(values, out);
    else
      SerializeRow(values, &out);
    engine_->Put(
        ToKey(key), std::string_view(out.data(), out.size()), ThreadTid());
    return kOK;
  }

  Status BatchInsert(const std::string&,
                     const std::vector<std::string>& keys,
                     std::vector<std::vector<Field>>& values) override {
    if (!engine_)
      return kError;
    if (keys.size() != values.size())
      return kError;
    std::vector<uint64_t> numeric_keys;
    numeric_keys.reserve(keys.size());
    std::vector<std::string> serialized;
    serialized.reserve(keys.size());
    std::vector<base::ConstArray<float>> batch_values;
    batch_values.reserve(keys.size());
    for (size_t i = 0; i < keys.size(); ++i) {
      numeric_keys.push_back(ToKey(keys[i]));
      serialized.emplace_back();
      if (embedding_mode_)
        SerializeEmbedding(values[i], serialized.back());
      else
        SerializeRow(values[i], &serialized.back());
      size_t rem = serialized.back().size() % sizeof(float);
      if (rem != 0) {
        serialized.back().resize(serialized.back().size() + sizeof(float) - rem, '\0');
      }
      batch_values.emplace_back(
          reinterpret_cast<const float*>(serialized.back().data()),
          serialized.back().size() / sizeof(float));
    }
    engine_->BatchPut(base::ConstArray<uint64_t>(numeric_keys), &batch_values, ThreadTid());
    return kOK;
  }

  Status Delete(const std::string&, const std::string& key) override {
    if (!engine_)
      return kError;
    engine_->Put(ToKey(key), std::string_view(), ThreadTid()); // 简易“删除”
    return kOK;
  }

private:
  static void SerializeEmbedding(
      const std::vector<Field>& values, std::string& out) {
    size_t total = 0;
    for (const auto& field : values)
      total += field.value.size();
    out.clear();
    out.reserve(total);
    for (const auto& field : values)
      out.append(field.value);
  }

  static void ReturnEmbedding(std::vector<Field>& result,
                              const std::string& blob) {
    result.clear();
    result.push_back({EMBEDDING_FIELD_NAME, blob});
  }

  static void SerializeRow(const std::vector<Field>& values, std::string* out) {
    // Build body: [len name][len value]...
    std::string body;
    body.reserve(256);
    for (const auto& f : values) {
      uint32_t nlen = static_cast<uint32_t>(f.name.size());
      uint32_t vlen = static_cast<uint32_t>(f.value.size());
      body.append(reinterpret_cast<const char*>(&nlen), sizeof(uint32_t));
      body.append(f.name.data(), f.name.size());
      body.append(reinterpret_cast<const char*>(&vlen), sizeof(uint32_t));
      body.append(f.value.data(), f.value.size());
    }
    // Prefix payload_len (without padding)
    out->clear();
    uint32_t payload_len = static_cast<uint32_t>(body.size());
    out->append(reinterpret_cast<const char*>(&payload_len), sizeof(uint32_t));
    out->append(body);
    // Pad to fixed value_size_ bytes if configured (>0)
    if (value_size_ > 0 && out->size() < value_size_) {
      out->resize(value_size_, '\0');
    }
  }

  static void DeserializeRowFilter(
      std::vector<Field>* values,
      const char* p,
      size_t n,
      const std::vector<std::string>& fields) {
    values->clear();
    if (n < sizeof(uint32_t))
      return;
    uint32_t payload_len = *reinterpret_cast<const uint32_t*>(p);
    p += sizeof(uint32_t);
    if (payload_len > n - sizeof(uint32_t))
      payload_len = static_cast<uint32_t>(n - sizeof(uint32_t));
    const char* lim = p + payload_len;

    auto it = fields.begin();
    while (p < lim && it != fields.end()) {
      if (p + sizeof(uint32_t) > lim)
        break;
      uint32_t nlen = *reinterpret_cast<const uint32_t*>(p);
      p += 4;
      if (p + nlen > lim)
        break;
      std::string name(p, nlen);
      p += nlen;
      if (p + sizeof(uint32_t) > lim)
        break;
      uint32_t vlen = *reinterpret_cast<const uint32_t*>(p);
      p += 4;
      if (p + vlen > lim)
        break;
      std::string val(p, vlen);
      p += vlen;
      if (*it == name) {
        values->push_back({name, val});
        ++it;
      }
    }
  }

  static void
  DeserializeRow(std::vector<Field>* values, const char* p, size_t n) {
    values->clear();
    if (n < sizeof(uint32_t))
      return;
    uint32_t payload_len = *reinterpret_cast<const uint32_t*>(p);
    p += sizeof(uint32_t);
    if (payload_len > n - sizeof(uint32_t))
      payload_len = static_cast<uint32_t>(n - sizeof(uint32_t));
    const char* lim = p + payload_len;
    while (p < lim) {
      if (p + sizeof(uint32_t) > lim)
        break;
      uint32_t nlen = *reinterpret_cast<const uint32_t*>(p);
      p += 4;
      if (p + nlen > lim)
        break;
      std::string name(p, nlen);
      p += nlen;
      if (p + sizeof(uint32_t) > lim)
        break;
      uint32_t vlen = *reinterpret_cast<const uint32_t*>(p);
      p += 4;
      if (p + vlen > lim)
        break;
      std::string val(p, vlen);
      p += vlen;
      values->push_back({name, val});
    }
  }

private:
  static inline KVEngineCCEH* engine_ = nullptr;
  static inline size_t field_cnt_     = 0;
  static inline size_t value_size_    = 0;
  static inline bool embedding_mode_  = false;
  static inline int ref_cnt_          = 0;
  static inline std::mutex mu_;
};

static DB* NewCCEH() { return new CCEHDB(); }
// 注册到 YCSB
const bool registered_cceh = ycsbc::DBFactory::RegisterDB("cceh", NewCCEH);

} // namespace ycsbc
