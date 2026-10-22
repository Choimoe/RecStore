#include <algorithm>
#include <cstdint>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <functional>
#include <stdexcept>
#include <string>
#include <vector>

#include <gtest/gtest.h>
#include <unistd.h>

#include "storage/ssd_layout/ssd_layout.h"

namespace {

class LayoutTest : public ::testing::Test {
protected:
  void SetUp() override {
    root_ = std::filesystem::temp_directory_path() /
            ("recstore-ssd-layout-" + std::to_string(getpid()));
    std::filesystem::remove_all(root_);
    std::filesystem::create_directories(root_);
  }
  void TearDown() override { std::filesystem::remove_all(root_); }

  ssd_layout::Hypergraph WriteLogs() {
    text_ = root_ / "query.txt";
    binary_ = root_ / "query.bin";
    std::ofstream text(text_);
    text << "0 4 3 6\n0 1\n2 3\n1 2\n";
    text.close();

    const uint64_t queries = 3;
    const uint64_t vertices = 4;
    const uint64_t index[] = {0, 2, 4, 6};
    const uint32_t pins[] = {0, 1, 2, 3, 1, 2};
    std::ofstream binary(binary_, std::ios::binary);
    binary.write(reinterpret_cast<const char*>(&queries), sizeof(queries));
    binary.write(reinterpret_cast<const char*>(&vertices), sizeof(vertices));
    binary.write(reinterpret_cast<const char*>(index), sizeof(index));
    binary.write(reinterpret_cast<const char*>(pins), sizeof(pins));
    binary.close();
    return ssd_layout::ReadQueryLog(text_.string(), false);
  }

  std::filesystem::path root_;
  std::filesystem::path text_;
  std::filesystem::path binary_;
};

TEST_F(LayoutTest, TextAndBinaryLogsMatch) {
  WriteLogs();
  const auto text = ssd_layout::ReadQueryLog(text_.string(), false);
  const auto binary = ssd_layout::ReadQueryLog(binary_.string(), true);
  EXPECT_EQ(text.vertex_count, binary.vertex_count);
  EXPECT_EQ(text.xpins, binary.xpins);
  EXPECT_EQ(text.pins, binary.pins);
}

TEST_F(LayoutTest, PartitionIsDeterministicAndBounded) {
  const auto graph = WriteLogs();
  const auto keys = ssd_layout::ReadKeyMap("", graph.vertex_count);
  const auto first = ssd_layout::Partition(graph, keys, 4096, 2048);
  const auto second = ssd_layout::Partition(graph, keys, 4096, 2048);
  ASSERT_EQ(first.keys.size(), 4U);
  EXPECT_EQ(first.keys, second.keys);
  EXPECT_EQ(first.copies, second.copies);
  for (const auto& row : first.copies) {
    ASSERT_EQ(row.size(), 1U);
    EXPECT_LE(row[0].byte_offset, 2048U);
  }
  for (const auto& page : first.pages) {
    EXPECT_LE(page.size(), 2U);
    EXPECT_FALSE(page.empty());
  }
  const auto path = root_ / "bandana.bin";
  ssd_layout::SaveArtifact(first, path.string());
  const auto loaded = ssd_layout::LoadArtifact(path.string());
  EXPECT_NE(loaded.checksum, 0U);
  EXPECT_EQ(loaded.keys, first.keys);
  EXPECT_EQ(loaded.copies, first.copies);
  EXPECT_THROW(ssd_layout::CheckArtifact(loaded, ssd_layout::Strategy::kMaxEmbed, 2048),
               std::invalid_argument);
}

TEST_F(LayoutTest, ReplicationStaysInBudgetAndShrinkKeepsPrimary) {
  const auto graph = WriteLogs();
  const auto keys = ssd_layout::ReadKeyMap("", graph.vertex_count);
  const auto bandana = ssd_layout::Partition(graph, keys, 4096, 2048);
  const auto replicated = ssd_layout::Replicate(bandana, graph, keys, 1.0);
  EXPECT_EQ(replicated.strategy, ssd_layout::Strategy::kMaxEmbed);
  uint64_t copies = 0;
  bool beneficial = false;
  for (size_t i = 0; i < replicated.copies.size(); ++i) {
    copies += replicated.copies[i].size();
    EXPECT_EQ(replicated.copies[i].front().page_id, bandana.copies[i].front().page_id);
  }
  EXPECT_LE(copies, 8U);
  for (const auto& page : replicated.pages) {
    std::vector<uint64_t> primaries;
    for (const auto& item : page) {
      const auto row = std::find(replicated.keys.begin(), replicated.keys.end(), item.first);
      primaries.push_back(replicated.copies[row - replicated.keys.begin()].front().page_id);
    }
    if (!primaries.empty() &&
        std::adjacent_find(primaries.begin(), primaries.end(), std::not_equal_to<uint64_t>()) !=
            primaries.end()) {
      beneficial = true;
    }
  }
  EXPECT_TRUE(beneficial);

  const auto shrunk = ssd_layout::Shrink(replicated, 0.0);
  ASSERT_EQ(shrunk.copies.size(), bandana.copies.size());
  for (size_t i = 0; i < shrunk.copies.size(); ++i) {
    ASSERT_EQ(shrunk.copies[i].size(), 1U);
    EXPECT_EQ(shrunk.copies[i].front().page_id, bandana.copies[i].front().page_id);
  }
}

TEST_F(LayoutTest, RejectsCorruptArtifact) {
  const auto graph = WriteLogs();
  const auto keys = ssd_layout::ReadKeyMap("", graph.vertex_count);
  const auto path = root_ / "broken.bin";
  ssd_layout::SaveArtifact(ssd_layout::Partition(graph, keys, 4096, 128), path.string());
  std::fstream file(path, std::ios::binary | std::ios::in | std::ios::out);
  file.seekp(-1, std::ios::end);
  char byte = 0;
  file.seekg(-1, std::ios::end);
  file.read(&byte, 1);
  byte ^= 0x1;
  file.seekp(-1, std::ios::end);
  file.write(&byte, 1);
  file.close();
  EXPECT_THROW(ssd_layout::LoadArtifact(path.string()), std::invalid_argument);
  EXPECT_THROW(ssd_layout::ParseStrategy("mystery"), std::invalid_argument);
}

} // namespace
