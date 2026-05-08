#include <filesystem>
#include <fstream>
#include <memory>
#include <string>
#include <vector>

#include <gtest/gtest.h>
#include <unistd.h>

#include "base/factory.h"
#include "base/json.h"
#include "storage/kv_engine/base_kv.h"
#include "storage/kv_engine/engine_factory.h"
#include "storage/kv_engine/engine_selector.h"

namespace {

constexpr size_t kValueSize = 128;

BaseKVConfig MakeExternalEngineConfig(const std::string& engine_type,
                                      const std::string& path) {
  BaseKVConfig config;
  config.num_threads_ = 4;
  config.json_config_ = {
      {"engine_type", engine_type},
      {"path", path},
      {"capacity", 1024},
      {"value_size", kValueSize},
      {"max_batch_size", 16},
      {"index", {{"type", "DRAM_EXTENDIBLE_HASH"}}},
      {"value",
       {{"type", "DRAM_VALUE_STORE"},
        {"default_value_size_hint", kValueSize},
        {"dram_allocator",
         {{"type", "PERSIST_LOOP_SLAB"},
          {"capacity_bytes", 1024 * kValueSize}}}}}};
  return config;
}

std::unique_ptr<BaseKV> CreateEngine(const std::string& engine_type) {
  const std::string path = "/tmp/test_external_kv_engine_" + engine_type + "_" +
                           std::to_string(static_cast<long long>(getpid()));
  std::filesystem::remove_all(path);
  std::filesystem::create_directories(path);

  BaseKVConfig config = MakeExternalEngineConfig(engine_type, path);
  base::EngineResolved resolved;
  EXPECT_NO_THROW(resolved = base::ResolveEngine(config));
  EXPECT_EQ(resolved.engine, engine_type);
  return std::unique_ptr<BaseKV>(
      base::Factory<BaseKV, const BaseKVConfig&>::NewInstance(
          resolved.engine, resolved.cfg));
}

std::unique_ptr<BaseKV>
CreateEngineFromRecstoreConfigFile(const std::string& engine_type) {
  const std::string path =
      "/tmp/test_external_kv_engine_config_" + engine_type + "_" +
      std::to_string(static_cast<long long>(getpid()));
  const std::string config_path = path + ".json";
  std::filesystem::remove_all(path);
  std::filesystem::create_directories(path);

  json recstore_config = {
      {"cache_ps",
       {{"num_threads", 4},
        {"ps_type", "GRPC"},
        {"base_kv_config",
         {{"engine_type", engine_type},
          {"path", path},
          {"capacity", 1024},
          {"value_size", kValueSize},
          {"max_batch_size", 16},
          {"table_name", "default"},
          {"index", {{"type", "DRAM_EXTENDIBLE_HASH"}}},
          {"value",
           {{"type", "DRAM_VALUE_STORE"},
            {"default_value_size_hint", kValueSize},
            {"dram_allocator",
             {{"type", "PERSIST_LOOP_SLAB"},
              {"capacity_bytes", 1024 * kValueSize}}}}}}}}}};

  {
    std::ofstream out(config_path);
    out << recstore_config.dump(2);
  }

  std::ifstream in(config_path);
  json loaded;
  in >> loaded;

  BaseKVConfig config;
  config.num_threads_ = loaded.at("cache_ps").at("num_threads").get<int>();
  config.json_config_ = loaded.at("cache_ps").at("base_kv_config");

  base::EngineResolved resolved;
  EXPECT_NO_THROW(resolved = base::ResolveEngine(config));
  EXPECT_EQ(resolved.engine, engine_type);
  return std::unique_ptr<BaseKV>(
      base::Factory<BaseKV, const BaseKVConfig&>::NewInstance(
          resolved.engine, resolved.cfg));
}

void AssertBasicPutGet(BaseKV* kv) {
  std::string value = "external_engine_value";
  value.resize(kValueSize, '\0');

  kv->Put(7, value, 0);

  std::string out;
  kv->Get(7, out, 0);
  EXPECT_EQ(out, value);

  std::string miss;
  kv->Get(99, miss, 0);
  EXPECT_TRUE(miss.empty());
}

void AssertBatchPutGet(BaseKV* kv) {
  constexpr int kRows = 3;
  constexpr int kDim  = static_cast<int>(kValueSize / sizeof(float));

  std::vector<uint64_t> keys = {101, 102, 103};
  std::vector<std::vector<float>> rows(kRows, std::vector<float>(kDim));
  std::vector<base::ConstArray<float>> views;
  views.reserve(kRows);
  for (int i = 0; i < kRows; ++i) {
    for (int j = 0; j < kDim; ++j) {
      rows[i][j] = static_cast<float>(i * 100 + j);
    }
    views.emplace_back(rows[i].data(), rows[i].size());
  }

  base::ConstArray<uint64_t> key_view(keys.data(), keys.size());
  kv->BatchPut(key_view, &views, 0);

  std::vector<base::ConstArray<float>> out;
  kv->BatchGet(key_view, &out, 0);

  ASSERT_EQ(out.size(), keys.size());
  for (int i = 0; i < kRows; ++i) {
    ASSERT_EQ(out[i].Size(), kDim);
    for (int j = 0; j < kDim; ++j) {
      EXPECT_FLOAT_EQ(out[i][j], rows[i][j]);
    }
  }
}

void AssertFactoryEngine(const std::string& engine_type) {
  auto kv = CreateEngine(engine_type);
  ASSERT_NE(kv, nullptr);
  AssertBasicPutGet(kv.get());
  AssertBatchPutGet(kv.get());
}

void AssertConfigFileEngine(const std::string& engine_type) {
  auto kv = CreateEngineFromRecstoreConfigFile(engine_type);
  ASSERT_NE(kv, nullptr);
  AssertBasicPutGet(kv.get());
}

} // namespace

#ifdef RECSTORE_TEST_ENABLE_FASTERKV_ENGINE
TEST(ExternalKVEngineFactoryTest, FasterKVEngineUsesBaseKVInterface) {
  AssertFactoryEngine("KVEngineFasterKV");
}

TEST(ExternalKVEngineFactoryTest, FasterKVEngineCanBeSelectedByConfigFile) {
  AssertConfigFileEngine("KVEngineFasterKV");
}
#endif

#ifdef RECSTORE_TEST_ENABLE_HPS_ENGINE
TEST(ExternalKVEngineFactoryTest, HpsHashMapEngineUsesBaseKVInterface) {
  AssertFactoryEngine("KVEngineHPSHashMap");
}

TEST(ExternalKVEngineFactoryTest, HpsRocksDBEngineUsesBaseKVInterface) {
  AssertFactoryEngine("KVEngineHPSRocksDB");
}

TEST(ExternalKVEngineFactoryTest, HpsHashMapEngineCanBeSelectedByConfigFile) {
  AssertConfigFileEngine("KVEngineHPSHashMap");
}

TEST(ExternalKVEngineFactoryTest, HpsRocksDBEngineCanBeSelectedByConfigFile) {
  AssertConfigFileEngine("KVEngineHPSRocksDB");
}
#endif
