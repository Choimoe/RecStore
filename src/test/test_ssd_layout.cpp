#include <algorithm>
#include <chrono>
#include <cstdint>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <functional>
#include <limits>
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

TEST_F(LayoutTest, TextLogNormalizesStartingOffset) {
  const auto path = root_ / "offset-query.txt";
  std::ofstream(path) << "100 4 2 4\n100 103\n101 102\n";
  const auto graph = ssd_layout::ReadQueryLog(path.string(), false);
  EXPECT_EQ(graph.vertex_count, 4U);
  EXPECT_EQ(graph.xpins, (std::vector<uint64_t>{0, 2, 4}));
  EXPECT_EQ(graph.pins, (std::vector<uint32_t>{0, 3, 1, 2}));
}

TEST_F(LayoutTest, RejectsQueryCountsOutsideUint32Range) {
  constexpr uint64_t too_many =
      static_cast<uint64_t>(std::numeric_limits<uint32_t>::max()) + 1;
  const auto text_path = root_ / "too-many-queries.txt";
  std::ofstream(text_path) << "0 1 " << too_many << " 0\n";
  try {
    (void)ssd_layout::ReadQueryLog(text_path.string(), false);
    FAIL() << "oversized text query count was accepted";
  } catch (const std::invalid_argument& error) {
    EXPECT_STREQ(error.what(), "query log query count exceeds uint32 range");
  }

  const auto binary_path  = root_ / "too-many-queries.bin";
  const uint64_t vertices = 1;
  std::ofstream binary(binary_path, std::ios::binary);
  binary.write(reinterpret_cast<const char*>(&too_many), sizeof(too_many));
  binary.write(reinterpret_cast<const char*>(&vertices), sizeof(vertices));
  binary.close();
  try {
    (void)ssd_layout::ReadQueryLog(binary_path.string(), true);
    FAIL() << "oversized binary query count was accepted";
  } catch (const std::invalid_argument& error) {
    EXPECT_STREQ(error.what(), "query log query count exceeds uint32 range");
  }
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

TEST_F(LayoutTest, PartitionScalesWithCurrentSubgraph) {
  constexpr uint32_t vertices  = 262144;
  constexpr uint32_t edge_size = 32;
  ssd_layout::Hypergraph graph;
  graph.vertex_count = vertices;
  graph.xpins.reserve(vertices / edge_size + 1);
  graph.pins.reserve(vertices);
  graph.xpins.push_back(0);
  for (uint32_t vertex = 0; vertex < vertices; ++vertex) {
    graph.pins.push_back(vertex);
    if ((vertex + 1) % edge_size == 0) {
      graph.xpins.push_back(graph.pins.size());
    }
  }

  const auto keys    = ssd_layout::ReadKeyMap("", graph.vertex_count);
  const auto begin   = std::chrono::steady_clock::now();
  const auto layout  = ssd_layout::Partition(graph, keys, 4096, 128);
  const auto elapsed = std::chrono::steady_clock::now() - begin;

  EXPECT_EQ(layout.keys.size(), vertices);
  EXPECT_LT(elapsed, std::chrono::seconds(3));
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

TEST_F(LayoutTest, ReplicationDoesNotRoundCopyBudgetUpToAFullPage) {
  const auto graph = WriteLogs();
  const auto keys = ssd_layout::ReadKeyMap("", graph.vertex_count);
  const auto bandana = ssd_layout::Partition(graph, keys, 4096, 2048);
  const auto replicated = ssd_layout::Replicate(bandana, graph, keys, 0.25);
  uint64_t copies = 0;
  for (const auto& row : replicated.copies) {
    copies += row.size();
  }
  EXPECT_LE(copies, 5U);
}

TEST_F(LayoutTest, ReplicationRejectsOutOfRangeVertices) {
  auto graph = WriteLogs();
  const auto keys = ssd_layout::ReadKeyMap("", graph.vertex_count);
  const auto bandana = ssd_layout::Partition(graph, keys, 4096, 2048);
  graph.pins[0] = static_cast<uint32_t>(graph.vertex_count);
  EXPECT_THROW(ssd_layout::Replicate(bandana, graph, keys, 1.0),
               std::invalid_argument);
}

TEST_F(LayoutTest, RejectsNonFiniteRatios) {
  const auto graph = WriteLogs();
  const auto keys = ssd_layout::ReadKeyMap("", graph.vertex_count);
  const auto bandana = ssd_layout::Partition(graph, keys, 4096, 2048);
  EXPECT_THROW(ssd_layout::Replicate(
                   bandana, graph, keys,
                   std::numeric_limits<double>::quiet_NaN()),
               std::invalid_argument);
  EXPECT_THROW(ssd_layout::Shrink(
                   bandana, std::numeric_limits<double>::infinity()),
               std::invalid_argument);
}

TEST_F(LayoutTest, RejectsUnknownArtifactStrategy) {
  const auto graph = WriteLogs();
  const auto keys = ssd_layout::ReadKeyMap("", graph.vertex_count);
  auto artifact = ssd_layout::Partition(graph, keys, 4096, 2048);
  artifact.strategy = static_cast<ssd_layout::Strategy>(99);
  EXPECT_THROW(ssd_layout::Normalize(std::move(artifact)),
               std::invalid_argument);
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
