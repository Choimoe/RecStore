#pragma once

#include <cstdint>
#include <string>
#include <utility>
#include <vector>

namespace ssd_layout {

inline constexpr uint64_t kMaxKey = (1ULL << 63) - 2;

enum class Strategy : uint32_t { kPlain = 0, kBandana = 1, kMaxEmbed = 2 };

struct Location {
  uint64_t page_id     = 0;
  uint32_t byte_offset = 0;
  bool operator==(const Location& other) const {
    return page_id == other.page_id && byte_offset == other.byte_offset;
  }
};

struct Hypergraph {
  uint64_t vertex_count = 0;
  std::vector<uint64_t> xpins;
  std::vector<uint32_t> pins;
};

struct Artifact {
  Strategy strategy  = Strategy::kBandana;
  uint32_t page_size = 4096;
  uint32_t row_size  = 0;
  uint64_t checksum  = 0;
  std::vector<uint64_t> keys;
  std::vector<std::vector<Location>> copies;
  std::vector<std::vector<std::pair<uint64_t, uint32_t>>> pages;
};

Strategy ParseStrategy(const std::string& name);
const char* StrategyName(Strategy strategy);

Hypergraph ReadQueryLog(const std::string& path, bool binary);
std::vector<uint64_t>
ReadKeyMap(const std::string& path, uint64_t vertex_count);

Artifact Partition(const Hypergraph& graph,
                   const std::vector<uint64_t>& keys,
                   uint32_t page_size,
                   uint32_t row_size);
Artifact Replicate(const Artifact& partitioned,
                   const Hypergraph& graph,
                   const std::vector<uint64_t>& keys,
                   double rep_ratio);
Artifact Shrink(const Artifact& replicated, double rep_ratio);

Artifact Normalize(Artifact artifact);
void SaveArtifact(const Artifact& artifact, const std::string& path);
Artifact LoadArtifact(const std::string& path);
void CheckArtifact(
    const Artifact& artifact, Strategy expected, uint32_t expected_row_size);

} // namespace ssd_layout
