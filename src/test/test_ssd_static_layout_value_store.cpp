#include <array>
#include <atomic>
#include <cstdint>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <stdexcept>
#include <string>
#include <thread>
#include <unordered_map>
#include <vector>

#include <gtest/gtest.h>
#include <unistd.h>

#include "memory/shm_file.h"
#include "storage/index/utils/hash.h"
#include "storage/io_backend/force_link.h"
#include "storage/io_backend/io_backend.h"
#include "storage/kv_engine/engine_composite.h"
#include "storage/ssd_layout/ssd_layout.h"
#include "storage/value_store/hybrid_value_store.h"
#include "storage/value_store/ssd_static_layout_value_store.h"
#include "storage/value_store/ssd_value_store.h"

namespace {

std::atomic<uint64_t> g_read_entries{0};

class CountingIOBackend : public IOBackend {
public:
  explicit CountingIOBackend(const BaseKVConfig& config) : IOBackend(config) {}
  void init() override {}
  void* GetPage(coroutine<void>::push_type&, uint64_t, PageID_t page_id) override {
    return GetPage(page_id);
  }
  void* GetPage(PageID_t page_id) override {
    auto* page = new char[PAGE_SIZE];
    ReadPageSync(page_id, page);
    return page;
  }
  void Unpin(coroutine<void>::push_type&, uint64_t, PageID_t page_id, void* page,
             bool dirty) override {
    Unpin(page_id, page, dirty);
  }
  void Unpin(PageID_t page_id, void* page_data, bool is_dirty) override {
    if (is_dirty) {
      WritePageSync(page_id, static_cast<char*>(page_data));
    }
    delete[] static_cast<char*>(page_data);
  }
  char* AllocateBuffer() override { return AllocateBuffer(1); }
  char* AllocateBuffer(uint64_t page_count) override {
    return new char[page_count * PAGE_SIZE]();
  }
  void FreeBuffer(char* buf) override { delete[] buf; }
  void submit() override {}
  void PollCompletion() override {}
  void BatchReadPages(const std::vector<IOEntry>& entries) override {
    g_read_entries.fetch_add(entries.size(), std::memory_order_relaxed);
    IOBackend::BatchReadPages(entries);
  }

protected:
  void ReadPageAsync(coroutine<void>::push_type&, uint64_t, PageID_t page_id,
                     char* buffer) override {
    ReadPageSync(page_id, buffer);
  }
  void ReadPageSync(PageID_t page_id, char* buffer) override {
    auto found = pages_.find(page_id);
    if (found == pages_.end()) {
      std::memset(buffer, 0, PAGE_SIZE);
      return;
    }
    std::memcpy(buffer, found->second.data(), PAGE_SIZE);
  }
  void WritePageAsync(coroutine<void>::push_type&, uint64_t, PageID_t page_id,
                      char* buffer) override {
    WritePageSync(page_id, buffer);
  }
  void WritePageSync(PageID_t page_id, char* buffer) override {
    auto& page = pages_[page_id];
    if (page.empty()) {
      page.resize(PAGE_SIZE, 0);
    }
    std::memcpy(page.data(), buffer, PAGE_SIZE);
  }

private:
  std::unordered_map<PageID_t, std::vector<char>> pages_;
};

FACTORY_REGISTER(IOBackend, INMEMORY, CountingIOBackend, const BaseKVConfig&);

class StaticLayoutValueTest : public ::testing::Test {
protected:
  void SetUp() override {
    root_ = std::filesystem::temp_directory_path() /
            ("recstore-static-value-" + std::to_string(getpid()));
    std::filesystem::remove_all(root_);
    std::filesystem::create_directories(root_);
    base::PMMmapRegisterCenter::GetConfig().backend =
        base::PMMmapRegisterCenter::Backend::kAnonymousDram;
    g_read_entries.store(0);
  }
  void TearDown() override { std::filesystem::remove_all(root_); }

  std::filesystem::path Save(const ssd_layout::Artifact& artifact) {
    const auto path = root_ / "layout.bin";
    ssd_layout::SaveArtifact(ssd_layout::Normalize(artifact), path.string());
    return path;
  }

  BaseKVConfig Config(const std::filesystem::path& mapping, const char* strategy,
                      const char* value_type = "SSD_STATIC_LAYOUT_VALUE_STORE") {
    BaseKVConfig config;
    config.num_threads_ = 2;
    config.json_config_ = {
        {"engine_type", "KVEngineComposite"},
        {"capacity", 16},
        {"index", {{"type", "DRAM_UNORDERED_MAP"}}},
        {"value",
         {{"type", value_type},
          {"path", (root_ / "value.db").string()},
          {"default_value_size_hint", 4},
          {"io", {{"type", "INMEMORY"}, {"queue_depth", 4}, {"base_offset_bytes", 0}}},
          {"ssd_layout",
           {{"strategy", strategy},
            {"mapping_path", mapping.string()}}}}}};
    return config;
  }

  std::filesystem::path root_;
};

ssd_layout::Artifact Bandana() {
  ssd_layout::Artifact artifact;
  artifact.strategy   = ssd_layout::Strategy::kBandana;
  artifact.page_size  = 4096;
  artifact.row_size   = 4;
  artifact.keys       = {0, 1, 2, 3, 4, 5};
  artifact.copies     = {{{0, 0}}, {{0, 4}}, {{0, 8}}, {{1, 0}}, {{1, 4}}, {{1, 8}}};
  return artifact;
}

ssd_layout::Artifact MaxEmbed() {
  auto artifact   = Bandana();
  artifact.strategy = ssd_layout::Strategy::kMaxEmbed;
  for (uint32_t slot = 0; slot < artifact.keys.size(); ++slot) {
    artifact.copies[slot].push_back({2, slot * 4});
  }
  return artifact;
}

TEST_F(StaticLayoutValueTest, KeyHandleRoundTripAndFixedPlacement) {
  auto store = std::make_unique<SsdStaticLayoutValueStore>(
      Config(Save(Bandana()), "bandana"));
  const char zero[] = {'z', 'e', 'r', 'o'};
  const char max_key_is_separate[] = {'n', 'o', 'p', 'e'};
  store->Write(1, zero, 4);
  std::string out(4, '\0');
  EXPECT_EQ(store->Read(1, out.data(), out.size()), 4U);
  EXPECT_EQ(out, std::string(zero, 4));
  store->Free(1);
  EXPECT_EQ(store->Read(1, out.data(), out.size()), 4U);
  EXPECT_THROW(store->Alloc(4), std::invalid_argument);
  EXPECT_THROW(store->AllocAndWrite(zero, 4), std::invalid_argument);
  EXPECT_THROW(store->Write(99, zero, 4), std::invalid_argument);
  EXPECT_THROW(store->Write(1ULL << 63, zero, 4), std::invalid_argument);
  EXPECT_THROW(store->Write(1, max_key_is_separate, 3), std::invalid_argument);
  EXPECT_THROW(SsdStaticLayoutValueStore(Config(Save(Bandana()), "plain")),
               std::invalid_argument);

  ssd_layout::Artifact wide = Bandana();
  wide.keys = {ssd_layout::kMaxKey};
  wide.copies = {{{0, 0}}};
  auto max_store = std::make_unique<SsdStaticLayoutValueStore>(
      Config(Save(wide), "bandana"));
  max_store->Write(ssd_layout::kMaxKey + 1, zero, 4);
  EXPECT_EQ(max_store->Read(ssd_layout::kMaxKey + 1, out.data(), out.size()), 4U);
}

TEST_F(StaticLayoutValueTest, BatchReadUsesFewerPagesThanPlain) {
  const char payload[] = {'a', 'b', 'c', 'd'};
  BaseKVConfig plain_config;
  plain_config.json_config_ = {
      {"value",
       {{"type", "SSD_VALUE_STORE"},
        {"path", (root_ / "plain.db").string()},
        {"ssd_allocator",
         {{"type", "SSD_SLAB"},
          {"capacity_bytes", 64ULL << 20},
          {"size_classes", {128}},
          {"io", {{"type", "INMEMORY"}, {"queue_depth", 4}, {"base_offset_bytes", 0}}}}}}}};
  SsdValueStore plain(plain_config);
  std::vector<uint64_t> plain_handles;
  for (int i = 0; i < 6; ++i) {
    plain_handles.push_back(plain.AllocAndWrite(payload, 4));
  }
  plain.ResetStats();
  g_read_entries.store(0);
  std::vector<ValueStore::ReadResult> plain_out;
  plain.BatchRead(plain_handles, plain_out);
  const uint64_t plain_reads = g_read_entries.load();
  EXPECT_NE(plain.ExtraResultFields().find("layout_page_reads=6"),
            std::string::npos);
  EXPECT_NE(plain.ExtraResultFields().find("layout_bytes_read=24576"),
            std::string::npos);

  auto bandana = std::make_unique<SsdStaticLayoutValueStore>(
      Config(Save(Bandana()), "bandana"));
  auto maxembed = std::make_unique<SsdStaticLayoutValueStore>(
      Config(Save(MaxEmbed()), "maxembed"));
  std::vector<uint64_t> handles;
  std::vector<ValueStore::WriteSpec> specs;
  for (uint64_t key = 0; key < 6; ++key) {
    handles.push_back(key + 1);
    specs.push_back({payload, 4});
  }
  bandana->BatchWrite(handles, specs);
  maxembed->BatchWrite(handles, specs);
  std::vector<ValueStore::ReadResult> bandana_out;
  std::vector<ValueStore::ReadResult> max_out;
  bandana->BatchRead(handles, bandana_out);
  maxembed->BatchRead(handles, max_out);
  EXPECT_LT(maxembed->stats().page_reads, bandana->stats().page_reads);
  EXPECT_LT(bandana->stats().page_reads, plain_reads);
  ASSERT_EQ(bandana_out.size(), 6U);
  EXPECT_EQ(bandana_out.front().data, std::vector<char>(payload, payload + 4));
  EXPECT_EQ(max_out.back().data, bandana_out.back().data);

  std::vector<uint64_t> repeated = {1, 1, 1};
  std::vector<ValueStore::ReadResult> repeated_out;
  const auto before = maxembed->stats().page_reads;
  maxembed->BatchRead(repeated, repeated_out);
  EXPECT_EQ(maxembed->stats().page_reads, before + 1);
  EXPECT_EQ(repeated_out[0].data, repeated_out[2].data);
}

TEST_F(StaticLayoutValueTest, HybridWatermarkAndReplicaUpdate) {
  auto mapping = Save(MaxEmbed());
  BaseKVConfig config = Config(mapping, "maxembed", "TIERED_VALUE_STORE");
  config.json_config_["value"]["dram_allocator"] = {
      {"type", "PERSIST_LOOP_SLAB"},
      {"capacity_bytes", 1 << 20},
      {"path", (root_ / "dram").string()}};
  config.json_config_["value"]["ssd_allocator"] = {
      {"type", "SSD_SLAB"},
      {"capacity_bytes", 64ULL << 20},
      {"path", (root_ / "ssd.db").string()},
      {"io",
       {{"type", "INMEMORY"}, {"queue_depth", 4}, {"base_offset_bytes", 0}}}};
  config.json_config_["value"]["tiering"] = {{"high_watermark_ratio", 0.0}};
  KVEngineComposite engine(config);
  for (uint64_t key = 0; key < 6; ++key) {
    engine.Put(key, std::string(4, static_cast<char>('A' + key)), 0);
  }
  const std::string updated(4, 'Z');
  engine.Put(2, updated, 0);
  std::string actual;
  engine.Get(2, actual, 0);
  EXPECT_EQ(actual, updated);
  const std::vector<uint64_t> keys = {0, 1, 2, 3, 4, 5};
  std::vector<base::ConstArray<float>> values;
  engine.BatchGet(base::ConstArray<uint64_t>(keys), &values, 0);
  ASSERT_EQ(values.size(), keys.size());
  EXPECT_EQ(std::string(reinterpret_cast<const char*>(values[2].Data()), 4), updated);

  config.json_config_["value"]["tiering"] = {
      {"high_watermark_ratio", 4.0 / static_cast<double>(1 << 20)}};
  HybridValueStore hybrid(config);
  const char payload[4] = {'a', 'a', 'a', 'a'};
  const uint64_t first  = hybrid.AllocAndWrite(0, payload, 4);
  const uint64_t second = hybrid.AllocAndWrite(1, payload, 4);
  EXPECT_EQ(first & (1ULL << 63), 0U);
  EXPECT_NE(second & (1ULL << 63), 0U);
}

TEST_F(StaticLayoutValueTest, HybridValidatesStaticRowsBeforeDramPlacement) {
  BaseKVConfig config =
      Config(Save(Bandana()), "bandana", "TIERED_VALUE_STORE");
  config.json_config_["value"]["dram_allocator"] = {
      {"type", "PERSIST_LOOP_SLAB"},
      {"capacity_bytes", 1 << 20},
      {"path", (root_ / "dram").string()}};
  config.json_config_["value"]["ssd_allocator"] = {
      {"type", "SSD_SLAB"},
      {"capacity_bytes", 64ULL << 20},
      {"path", (root_ / "ssd.db").string()},
      {"io",
       {{"type", "INMEMORY"}, {"queue_depth", 4}, {"base_offset_bytes", 0}}}};
  config.json_config_["value"]["tiering"] = {{"high_watermark_ratio", 1.0}};
  HybridValueStore hybrid(config);
  const char payload[4] = {'a', 'b', 'c', 'd'};
  EXPECT_THROW(hybrid.AllocAndWrite(99, payload, sizeof(payload)),
               std::invalid_argument);
  EXPECT_THROW(hybrid.AllocAndWrite(0, payload, sizeof(payload) - 1),
               std::invalid_argument);
}

TEST_F(StaticLayoutValueTest, StaticStoreUsesIoDefaultsWhenConfigIsOmitted) {
  BaseKVConfig config = Config(Save(Bandana()), "bandana");
  config.json_config_["value"].erase("io");
  ForceLinkIOBackends();
  EXPECT_NO_THROW({ SsdStaticLayoutValueStore store(config); });
}

TEST_F(StaticLayoutValueTest, BatchGetFlatUsesGroupedStaticReads) {
  const auto mapping               = Save(Bandana());
  const std::vector<uint64_t> keys = {0, 1, 2};
  const std::array<float, 3> input = {1.0F, 2.0F, 3.0F};
  auto exercise                    = [&](BaseKVConfig config) {
    KVEngineComposite engine(config);
    for (size_t i = 0; i < keys.size(); ++i) {
      engine.Put(
          keys[i],
          std::string(reinterpret_cast<const char*>(&input[i]), sizeof(float)),
          0);
    }
    g_read_entries.store(0);
    std::array<float, 3> output{};
    EXPECT_TRUE(engine.BatchGetFlat(
        base::ConstArray<uint64_t>(keys), output.data(), 3, 1, 0));
    EXPECT_EQ(output, input);
    EXPECT_EQ(g_read_entries.load(), 1U);
  };

  exercise(Config(mapping, "bandana"));

  BaseKVConfig hybrid = Config(mapping, "bandana", "TIERED_VALUE_STORE");
  hybrid.json_config_["value"]["dram_allocator"] = {
      {"type", "PERSIST_LOOP_SLAB"},
      {"capacity_bytes", 1 << 20},
      {"path", (root_ / "flat-dram").string()}};
  hybrid.json_config_["value"]["ssd_allocator"] = {
      {"type", "SSD_SLAB"},
      {"capacity_bytes", 64ULL << 20},
      {"path", (root_ / "flat-ssd.db").string()},
      {"io",
       {{"type", "INMEMORY"}, {"queue_depth", 4}, {"base_offset_bytes", 0}}}};
  hybrid.json_config_["value"]["tiering"] = {{"high_watermark_ratio", 0.0}};
  exercise(std::move(hybrid));
}

TEST_F(StaticLayoutValueTest, ConcurrentPageUpdatesDoNotTear) {
  auto store = std::make_unique<SsdStaticLayoutValueStore>(
      Config(Save(Bandana()), "bandana"));
  std::vector<std::thread> threads;
  for (int thread = 0; thread < 2; ++thread) {
    threads.emplace_back([&, thread] {
      const std::string value(4, static_cast<char>('A' + thread));
      for (int i = 0; i < 100; ++i) {
        store->Write(static_cast<uint64_t>(thread + 1), value.data(), value.size());
      }
    });
  }
  for (auto& thread : threads) {
    thread.join();
  }
  std::string left(4, '\0');
  std::string right(4, '\0');
  store->Read(1, left.data(), left.size());
  store->Read(2, right.data(), right.size());
  EXPECT_EQ(left, std::string(4, 'A'));
  EXPECT_EQ(right, std::string(4, 'B'));
}

void WriteLegacyCheckpoint(const std::filesystem::path& path,
                           const std::string& metadata) {
  const char magic[] = {'R', 'S', 'K', 'V', 'C', 'P', '0', '1'};
  const uint32_t version = 2;
  const uint64_t metadata_size = metadata.size();
  const uint64_t record_count = 0;
  uint64_t checksum = 0x9e3779b97f4a7c15ULL;
  auto mix = [&](const void* data, size_t size) {
    if (size != 0) {
      checksum = xxhash(data, size, checksum);
    }
  };
  mix(magic, sizeof(magic));
  mix(&version, sizeof(version));
  mix(&metadata_size, sizeof(metadata_size));
  mix(&record_count, sizeof(record_count));
  mix(metadata.data(), metadata.size());
  std::ofstream output(path, std::ios::binary | std::ios::trunc);
  output.write(magic, sizeof(magic));
  output.write(reinterpret_cast<const char*>(&version), sizeof(version));
  output.write(reinterpret_cast<const char*>(&metadata_size), sizeof(metadata_size));
  output.write(reinterpret_cast<const char*>(&record_count), sizeof(record_count));
  output.write(metadata.data(), metadata.size());
  output.write(reinterpret_cast<const char*>(&checksum), sizeof(checksum));
}

TEST_F(StaticLayoutValueTest, CheckpointIdentityRoundTripAndMismatch) {
  const std::string metadata = "layout-v3";
  const auto checkpoint = root_ / "layout.ckpt";
  auto source = std::make_unique<KVEngineComposite>(Config(Save(Bandana()), "bandana"));
  source->Put(0, std::string("abcd"), 0);
  source->Put(5, std::string("wxyz"), 0);
  ASSERT_TRUE(source->SaveCheckpoint(checkpoint.string(), metadata));

  auto restored = std::make_unique<KVEngineComposite>(Config(Save(Bandana()), "bandana"));
  ASSERT_TRUE(restored->LoadCheckpoint(checkpoint.string(), metadata));
  std::string actual;
  restored->Get(0, actual, 0);
  EXPECT_EQ(actual, "abcd");
  restored->Get(5, actual, 0);
  EXPECT_EQ(actual, "wxyz");

  auto mismatched = std::make_unique<KVEngineComposite>(
      Config(Save(MaxEmbed()), "maxembed"));
  EXPECT_FALSE(mismatched->LoadCheckpoint(checkpoint.string(), metadata));
  mismatched->Get(0, actual, 0);
  EXPECT_TRUE(actual.empty());

  const auto legacy = root_ / "legacy.ckpt";
  WriteLegacyCheckpoint(legacy, metadata);
  BaseKVConfig dram;
  dram.num_threads_ = 1;
  dram.json_config_ = {
      {"engine_type", "KVEngineComposite"},
      {"capacity", 16},
      {"index", {{"type", "DRAM_UNORDERED_MAP"}}},
      {"value",
       {{"type", "DRAM_VALUE_STORE"},
        {"path", (root_ / "dram-value").string()},
        {"default_value_size_hint", 4},
        {"dram_allocator",
         {{"type", "PERSIST_LOOP_SLAB"}, {"capacity_bytes", 1 << 20}}}}}};
  KVEngineComposite plain(dram);
  EXPECT_TRUE(plain.LoadCheckpoint(legacy.string(), metadata));
  auto rejected = std::make_unique<KVEngineComposite>(Config(Save(Bandana()), "bandana"));
  EXPECT_FALSE(rejected->LoadCheckpoint(legacy.string(), metadata));
  rejected->Get(0, actual, 0);
  EXPECT_TRUE(actual.empty());
}

} // namespace
