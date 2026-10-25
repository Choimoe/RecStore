#include "storage/ssd_layout/ssd_layout.h"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <fstream>
#include <limits>
#include <random>
#include <sstream>
#include <stdexcept>
#include <unordered_map>
#include <unordered_set>

namespace ssd_layout {
namespace {

constexpr char kMagic[8] = {'R', 'S', 'L', 'A', 'Y', '0', '0', '1'};
constexpr uint32_t kVersion = 1;
constexpr uint64_t kFnvOffset = 14695981039346656037ULL;
constexpr uint64_t kFnvPrime = 1099511628211ULL;
constexpr float kBalanceP = 0.5f;
constexpr int kMaxRefine = 8;

[[noreturn]] void Fail(const std::string& message) {
  throw std::invalid_argument(message);
}

uint64_t Fnv(const void* data, size_t size, uint64_t hash = kFnvOffset) {
  const auto* bytes = static_cast<const unsigned char*>(data);
  for (size_t i = 0; i < size; ++i) {
    hash ^= bytes[i];
    hash *= kFnvPrime;
  }
  return hash;
}

template <typename T>
uint64_t FnvPod(const T& value, uint64_t hash) {
  return Fnv(&value, sizeof(value), hash);
}

uint32_t SlotsPerPage(uint32_t page_size, uint32_t row_size) {
  if (page_size == 0 || row_size == 0 || row_size > page_size) {
    Fail("SSD layout row must fit in one page");
  }
  return page_size / row_size;
}

void CheckGraph(const Hypergraph& graph) {
  if (graph.vertex_count > std::numeric_limits<uint32_t>::max()) {
    Fail("SSD layout vertex count exceeds uint32");
  }
  if (graph.xpins.empty() || graph.xpins.front() != 0 ||
      graph.xpins.back() != graph.pins.size()) {
    Fail("query log index is invalid");
  }
  for (size_t i = 1; i < graph.xpins.size(); ++i) {
    if (graph.xpins[i] < graph.xpins[i - 1] ||
        graph.xpins[i] > graph.pins.size()) {
      Fail("query log edge range is invalid");
    }
  }
  for (uint32_t vertex : graph.pins) {
    if (vertex >= graph.vertex_count) {
      Fail("query log vertex is out of range");
    }
  }
}

void ReadAll(const std::string& path, std::vector<unsigned char>* out) {
  std::ifstream input(path, std::ios::binary);
  if (!input) {
    Fail("cannot open " + path);
  }
  input.seekg(0, std::ios::end);
  const auto size = input.tellg();
  if (size < 0) {
    Fail("cannot size " + path);
  }
  input.seekg(0);
  out->resize(static_cast<size_t>(size));
  if (!out->empty()) {
    input.read(reinterpret_cast<char*>(out->data()), size);
    if (!input) {
      Fail("truncated read " + path);
    }
  }
}

struct Rows {
  std::vector<uint64_t> keys;
  std::vector<std::vector<Location>> copies;
};

Rows RowsFromParts(const std::vector<uint64_t>& keys,
                   const std::vector<std::vector<uint32_t>>& pages_of_vertex,
                   uint32_t row_size) {
  Rows rows;
  rows.keys = keys;
  rows.copies.resize(keys.size());
  std::vector<std::pair<uint64_t, uint32_t>> order;
  order.reserve(keys.size());
  for (uint32_t vertex = 0; vertex < keys.size(); ++vertex) {
    order.push_back({keys[vertex], vertex});
  }
  std::sort(order.begin(), order.end());
  std::vector<std::vector<uint32_t>> members(1);
  for (uint32_t vertex = 0; vertex < pages_of_vertex.size(); ++vertex) {
    for (uint32_t page : pages_of_vertex[vertex]) {
      if (page >= members.size()) {
        members.resize(page + 1);
      }
      members[page].push_back(vertex);
    }
  }
  std::vector<std::unordered_map<uint32_t, uint32_t>> offset(members.size());
  for (uint32_t page = 0; page < members.size(); ++page) {
    auto& group = members[page];
    std::sort(group.begin(), group.end(), [&](uint32_t a, uint32_t b) {
      return keys[a] < keys[b];
    });
    group.erase(std::unique(group.begin(), group.end()), group.end());
    for (uint32_t slot = 0; slot < group.size(); ++slot) {
      offset[page][group[slot]] = slot * row_size;
    }
  }
  for (size_t row = 0; row < order.size(); ++row) {
    const uint32_t vertex = order[row].second;
    rows.keys[row]        = order[row].first;
    for (uint32_t page : pages_of_vertex[vertex]) {
      rows.copies[row].push_back(Location{page, offset[page].at(vertex)});
    }
  }
  return rows;
}

Artifact Finish(Strategy strategy,
                uint32_t page_size,
                uint32_t row_size,
                Rows rows) {
  if (strategy != Strategy::kPlain && strategy != Strategy::kBandana &&
      strategy != Strategy::kMaxEmbed) {
    Fail("unknown SSD layout strategy");
  }
  const uint32_t slots = SlotsPerPage(page_size, row_size);
  Artifact artifact;
  artifact.strategy   = strategy;
  artifact.page_size  = page_size;
  artifact.row_size   = row_size;
  artifact.keys       = std::move(rows.keys);
  artifact.copies     = std::move(rows.copies);
  if (artifact.keys.size() != artifact.copies.size()) {
    Fail("SSD layout key and copy counts differ");
  }
  std::unordered_set<uint64_t> seen_keys;
  std::unordered_set<uint64_t> seen_slots;
  uint32_t page_count = 0;
  for (size_t i = 0; i < artifact.keys.size(); ++i) {
    const uint64_t key = artifact.keys[i];
    if (i > 0 && key <= artifact.keys[i - 1]) {
      Fail("SSD layout keys must be strictly increasing");
    }
    if (key > kMaxKey || !seen_keys.insert(key).second) {
      Fail("SSD layout key is duplicated or out of range");
    }
    if (artifact.copies[i].empty()) {
      Fail("SSD layout key has no page");
    }
    for (const Location& copy : artifact.copies[i]) {
      if (copy.byte_offset / row_size >= slots ||
          copy.byte_offset % row_size != 0) {
        Fail("SSD layout slot is outside its page");
      }
      if (copy.page_id >= std::numeric_limits<uint32_t>::max()) {
        Fail("SSD layout page id exceeds uint32 range");
      }
      page_count = std::max(page_count, static_cast<uint32_t>(copy.page_id + 1));
      const uint64_t slot =
          (copy.page_id << 32) | static_cast<uint64_t>(copy.byte_offset);
      if (!seen_slots.insert(slot).second) {
        Fail("SSD layout slot is duplicated");
      }
    }
  }
  artifact.pages.resize(page_count);
  for (size_t i = 0; i < artifact.keys.size(); ++i) {
    for (const Location& copy : artifact.copies[i]) {
      artifact.pages[copy.page_id].push_back({artifact.keys[i], copy.byte_offset});
    }
  }
  for (auto& page : artifact.pages) {
    if (page.empty() || page.size() > slots) {
      Fail("SSD layout page is empty or over capacity");
    }
    std::sort(page.begin(), page.end());
  }
  return artifact;
}

std::vector<uint32_t> PartitionVertices(const Hypergraph& graph, uint32_t cap) {
  const uint64_t count = graph.vertex_count;
  std::vector<char> seen(count, 0);
  for (uint32_t vertex : graph.pins) {
    seen[vertex] = 1;
  }
  std::vector<uint32_t> hot;
  std::vector<uint32_t> cold;
  hot.reserve(count);
  for (uint32_t vertex = 0; vertex < count; ++vertex) {
    (seen[vertex] ? hot : cold).push_back(vertex);
  }
  std::mt19937 rng(1);
  std::shuffle(hot.begin(), hot.end(), rng);

  struct Edge {
    uint64_t begin;
    uint64_t end;
  };
  std::vector<Edge> edges;
  uint64_t longest = 1;
  for (uint64_t i = 0; i + 1 < graph.xpins.size(); ++i) {
    const uint64_t begin = graph.xpins[i];
    const uint64_t end   = graph.xpins[i + 1];
    if (end < begin || end > graph.pins.size()) {
      Fail("query log edge range is invalid");
    }
    if (end - begin >= 2) {
      edges.push_back({begin, end});
      longest = std::max(longest, end - begin);
    }
  }
  if (edges.size() > std::numeric_limits<uint32_t>::max()) {
    Fail("SSD layout edge count exceeds uint32");
  }
  std::vector<std::vector<uint32_t>> incident_edges(count);
  for (uint32_t edge = 0; edge < edges.size(); ++edge) {
    for (uint64_t pin = edges[edge].begin; pin < edges[edge].end; ++pin) {
      incident_edges[graph.pins[pin]].push_back(edge);
    }
  }
  std::vector<float> table(longest + 2);
  for (int i = -1; i <= static_cast<int>(longest); ++i) {
    table[static_cast<size_t>(i + 1)] = std::pow(1.f - kBalanceP, i);
  }
  const auto at = [&](int i) { return table[static_cast<size_t>(i + 1)]; };

  std::vector<uint32_t> idx = std::move(hot);
  std::vector<uint32_t> part(count, 0);
  std::vector<char> side(count, 0);
  std::vector<float> gain(count, 0);
  std::vector<uint32_t> span_epoch(count, 0);
  std::vector<uint32_t> edge_epoch(edges.size(), 0);
  uint32_t next_epoch = 1;
  uint32_t next_part = 0;

  // ponytail: adjacency costs O(vertices + pins) memory. Stream subgraphs if
  // traces outgrow RAM.
  const auto refine = [&](size_t begin, size_t mid, size_t end) {
    if (next_epoch == 0) {
      std::fill(span_epoch.begin(), span_epoch.end(), 0);
      std::fill(edge_epoch.begin(), edge_epoch.end(), 0);
      next_epoch = 1;
    }
    const uint32_t epoch = next_epoch++;
    for (size_t i = begin; i < mid; ++i) {
      side[idx[i]]       = 0;
      span_epoch[idx[i]] = epoch;
    }
    for (size_t i = mid; i < end; ++i) {
      side[idx[i]]       = 1;
      span_epoch[idx[i]] = epoch;
    }
    std::vector<uint32_t> relevant_edges;
    for (size_t i = begin; i < end; ++i) {
      for (uint32_t edge : incident_edges[idx[i]]) {
        if (edge_epoch[edge] != epoch) {
          edge_epoch[edge] = epoch;
          relevant_edges.push_back(edge);
        }
      }
    }
    std::sort(relevant_edges.begin(), relevant_edges.end());
    for (int iter = 0; iter < kMaxRefine; ++iter) {
      for (size_t i = begin; i < end; ++i) {
        gain[idx[i]] = 0.f;
      }
      for (uint32_t edge_id : relevant_edges) {
        const Edge& edge = edges[edge_id];
        int left = 0;
        int right = 0;
        for (uint64_t pin = edge.begin; pin < edge.end; ++pin) {
          const uint32_t vertex = graph.pins[pin];
          if (span_epoch[vertex] != epoch) {
            continue;
          }
          ++(side[vertex] == 0 ? left : right);
        }
        if (left + right < 2) {
          continue;
        }
        const float move_right = -(at(right) - at(left - 1));
        const float move_left  = -(at(left) - at(right - 1));
        for (uint64_t pin = edge.begin; pin < edge.end; ++pin) {
          const uint32_t vertex = graph.pins[pin];
          if (span_epoch[vertex] == epoch) {
            gain[vertex] += side[vertex] == 0 ? move_right : move_left;
          }
        }
      }
      std::sort(idx.begin() + static_cast<std::ptrdiff_t>(begin),
                idx.begin() + static_cast<std::ptrdiff_t>(mid),
                [&](uint32_t a, uint32_t b) { return gain[a] > gain[b]; });
      std::sort(idx.begin() + static_cast<std::ptrdiff_t>(mid),
                idx.begin() + static_cast<std::ptrdiff_t>(end),
                [&](uint32_t a, uint32_t b) { return gain[a] > gain[b]; });
      bool moved = false;
      for (size_t i = begin; i < mid; ++i) {
        const size_t other = mid + (i - begin);
        if (other >= end) {
          break;
        }
        if (gain[idx[i]] + gain[idx[other]] <= 0.f) {
          break;
        }
        std::swap(idx[i], idx[other]);
        side[idx[i]]     = 0;
        side[idx[other]] = 1;
        moved            = true;
      }
      if (!moved) {
        return;
      }
    }
  };

  const auto divide = [&](auto&& self, size_t begin, size_t end, uint32_t limit) -> void {
    if (end - begin <= limit) {
      const uint32_t id = next_part++;
      for (size_t i = begin; i < end; ++i) {
        part[idx[i]] = id;
      }
      return;
    }
    size_t mid = begin + ((end - begin) / (2 * limit)) * limit;
    if (end - begin < static_cast<size_t>(2) * limit) {
      mid = begin + (end - begin) / 2;
    }
    refine(begin, mid, end);
    self(self, begin, mid, limit);
    self(self, mid, end, limit);
  };
  if (!idx.empty()) {
    divide(divide, 0, idx.size(), cap);
  }
  uint32_t filled = cap;
  for (uint32_t vertex : cold) {
    if (filled == cap) {
      ++next_part;
      filled = 0;
    }
    part[vertex] = next_part - 1;
    ++filled;
  }
  return part;
}

std::vector<uint32_t> PrimaryPages(const Artifact& artifact,
                                   const std::vector<uint64_t>& keys) {
  std::unordered_map<uint64_t, uint32_t> page_of_key;
  page_of_key.reserve(artifact.keys.size());
  for (size_t i = 0; i < artifact.keys.size(); ++i) {
    page_of_key.emplace(artifact.keys[i],
                        static_cast<uint32_t>(artifact.copies[i].at(0).page_id));
  }
  std::vector<uint32_t> pages(keys.size());
  for (size_t vertex = 0; vertex < keys.size(); ++vertex) {
    const auto found = page_of_key.find(keys[vertex]);
    if (found == page_of_key.end()) {
      Fail("replication key is missing from the partitioned layout");
    }
    pages[vertex] = found->second;
  }
  return pages;
}

void AppendPod(std::vector<unsigned char>* out, const void* data, size_t size) {
  const auto* bytes = static_cast<const unsigned char*>(data);
  out->insert(out->end(), bytes, bytes + size);
}

template <typename T>
void Append(std::vector<unsigned char>* out, const T& value) {
  AppendPod(out, &value, sizeof(value));
}

template <typename T>
T ReadAt(const std::vector<unsigned char>& bytes, size_t* offset) {
  if (*offset + sizeof(T) > bytes.size()) {
    Fail("truncated SSD layout artifact");
  }
  T value;
  std::memcpy(&value, bytes.data() + *offset, sizeof(value));
  *offset += sizeof(value);
  return value;
}

} // namespace

Strategy ParseStrategy(const std::string& name) {
  if (name == "plain") {
    return Strategy::kPlain;
  }
  if (name == "bandana") {
    return Strategy::kBandana;
  }
  if (name == "maxembed") {
    return Strategy::kMaxEmbed;
  }
  Fail("unknown SSD layout strategy: " + name);
}

const char* StrategyName(Strategy strategy) {
  switch (strategy) {
  case Strategy::kPlain:
    return "plain";
  case Strategy::kBandana:
    return "bandana";
  case Strategy::kMaxEmbed:
    return "maxembed";
  }
  return "unknown";
}

Hypergraph ReadQueryLog(const std::string& path, bool binary) {
  std::vector<unsigned char> bytes;
  ReadAll(path, &bytes);
  Hypergraph graph;
  if (binary) {
    size_t offset = 0;
    const auto need = [&](size_t count, size_t width) {
      if (offset + count * width > bytes.size()) {
        Fail("truncated binary query log");
      }
    };
    need(2, sizeof(uint64_t));
    uint64_t queries = 0;
    uint64_t vertices = 0;
    std::memcpy(&queries, bytes.data() + offset, sizeof(queries));
    offset += sizeof(queries);
    std::memcpy(&vertices, bytes.data() + offset, sizeof(vertices));
    offset += sizeof(vertices);
    graph.vertex_count = vertices;
    graph.xpins.resize(queries + 1);
    need(queries + 1, sizeof(uint64_t));
    std::memcpy(graph.xpins.data(), bytes.data() + offset,
                (queries + 1) * sizeof(uint64_t));
    offset += (queries + 1) * sizeof(uint64_t);
    if (graph.xpins.empty() || graph.xpins.front() != 0) {
      Fail("binary query log index must start at zero");
    }
    for (size_t i = 1; i < graph.xpins.size(); ++i) {
      if (graph.xpins[i] < graph.xpins[i - 1]) {
        Fail("binary query log index is not monotonic");
      }
    }
    graph.pins.resize(graph.xpins.back());
    need(graph.pins.size(), sizeof(uint32_t));
    if (!graph.pins.empty()) {
      std::memcpy(graph.pins.data(), bytes.data() + offset,
                  graph.pins.size() * sizeof(uint32_t));
    }
    return graph;
  }

  std::ifstream input(path);
  if (!input) {
    Fail("cannot open " + path);
  }
  std::string line;
  if (!std::getline(input, line)) {
    Fail("text query log header is invalid");
  }
  std::istringstream header(line);
  uint64_t starting = 0;
  uint64_t vertices = 0;
  uint64_t queries = 0;
  uint64_t declared_pins = 0;
  if (!(header >> starting >> vertices >> queries >> declared_pins)) {
    Fail("text query log header is invalid");
  }
  graph.vertex_count = vertices;
  graph.xpins.resize(queries + 1);
  graph.pins.reserve(declared_pins);
  graph.xpins[0] = 0;
  for (uint64_t query = 0; query < queries; ++query) {
    if (!std::getline(input, line)) {
      Fail("text query log is missing a query");
    }
    std::istringstream row(line);
    std::unordered_set<uint32_t> unique;
    uint64_t value = 0;
    while (row >> value) {
      if (value < starting || value - starting >= vertices) {
        Fail("text query log vertex is out of range");
      }
      const auto vertex = static_cast<uint32_t>(value - starting);
      if (unique.insert(vertex).second) {
        graph.pins.push_back(vertex);
      }
    }
    graph.xpins[query + 1] = graph.pins.size();
  }
  if (graph.pins.size() > declared_pins) {
    Fail("text query log is longer than its header");
  }
  return graph;
}

std::vector<uint64_t> ReadKeyMap(const std::string& path, uint64_t vertex_count) {
  std::vector<uint64_t> keys(vertex_count);
  if (path.empty()) {
    for (uint64_t vertex = 0; vertex < vertex_count; ++vertex) {
      keys[vertex] = vertex;
    }
    return keys;
  }
  std::ifstream input(path);
  if (!input) {
    Fail("cannot open key map " + path);
  }
  for (uint64_t vertex = 0; vertex < vertex_count; ++vertex) {
    if (!(input >> keys[vertex]) || keys[vertex] > kMaxKey) {
      Fail("key map is short or out of range");
    }
  }
  std::unordered_set<uint64_t> unique(keys.begin(), keys.end());
  if (unique.size() != keys.size()) {
    Fail("key map contains a duplicate key");
  }
  return keys;
}

Artifact Partition(const Hypergraph& graph,
                   const std::vector<uint64_t>& keys,
                   uint32_t page_size,
                   uint32_t row_size) {
  CheckGraph(graph);
  if (keys.size() != graph.vertex_count) {
    Fail("key map size does not match the query log");
  }
  const uint32_t cap = SlotsPerPage(page_size, row_size);
  const std::vector<uint32_t> part = PartitionVertices(graph, cap);
  std::vector<std::vector<uint32_t>> pages(keys.size());
  for (uint32_t vertex = 0; vertex < keys.size(); ++vertex) {
    pages[vertex].push_back(part[vertex]);
  }
  return Finish(Strategy::kBandana, page_size, row_size,
                RowsFromParts(keys, pages, row_size));
}

Artifact Replicate(const Artifact& partitioned,
                   const Hypergraph& graph,
                   const std::vector<uint64_t>& keys,
                   double rep_ratio) {
  if (!std::isfinite(rep_ratio) || rep_ratio < 0.0) {
    Fail("replication ratio must be finite and non-negative");
  }
  CheckGraph(graph);
  if (keys.size() != graph.vertex_count || partitioned.keys.size() != keys.size()) {
    Fail("replication inputs do not describe the same vertices");
  }
  auto pages = std::vector<std::vector<uint32_t>>(keys.size());
  const std::vector<uint32_t> primary = PrimaryPages(partitioned, keys);
  uint32_t page_count = 0;
  for (uint32_t page : primary) {
    page_count = std::max(page_count, page + 1);
  }
  for (uint32_t vertex = 0; vertex < keys.size(); ++vertex) {
    pages[vertex].push_back(primary[vertex]);
  }
  if (rep_ratio == 0.0 || keys.empty()) {
    return Finish(Strategy::kMaxEmbed, partitioned.page_size, partitioned.row_size,
                  RowsFromParts(keys, pages, partitioned.row_size));
  }

  const uint32_t cap = SlotsPerPage(partitioned.page_size, partitioned.row_size);
  const long double requested =
      static_cast<long double>(rep_ratio) * keys.size();
  if (requested > std::numeric_limits<uint64_t>::max()) {
    Fail("replication ratio is too large");
  }
  const uint64_t extra = static_cast<uint64_t>(std::ceil(requested));
  const uint64_t replica_pages =
      extra / cap + static_cast<uint64_t>(extra % cap != 0);
  std::vector<uint64_t> score(keys.size(), 0);
  std::vector<std::vector<uint32_t>> edges;
  for (uint64_t edge = 0; edge + 1 < graph.xpins.size(); ++edge) {
    std::vector<uint32_t> vertices(
        graph.pins.begin() + static_cast<std::ptrdiff_t>(graph.xpins[edge]),
        graph.pins.begin() + static_cast<std::ptrdiff_t>(graph.xpins[edge + 1]));
    std::sort(vertices.begin(), vertices.end());
    vertices.erase(std::unique(vertices.begin(), vertices.end()), vertices.end());
    if (vertices.size() < 2) {
      continue;
    }
    std::unordered_set<uint32_t> parts;
    for (uint32_t vertex : vertices) {
      parts.insert(primary[vertex]);
    }
    if (parts.size() > 1) {
      for (uint32_t vertex : vertices) {
        score[vertex] += parts.size() - 1;
      }
    }
    edges.push_back(std::move(vertices));
  }
  std::vector<uint32_t> ranked(keys.size());
  for (uint32_t vertex = 0; vertex < ranked.size(); ++vertex) {
    ranked[vertex] = vertex;
  }
  std::sort(ranked.begin(), ranked.end(), [&](uint32_t a, uint32_t b) {
    return score[a] > score[b] || (score[a] == score[b] && a < b);
  });
  std::vector<std::vector<uint32_t>> node_edges(keys.size());
  for (uint32_t edge = 0; edge < edges.size(); ++edge) {
    for (uint32_t vertex : edges[edge]) {
      node_edges[vertex].push_back(edge);
    }
  }
  uint64_t remaining = extra;
  for (uint64_t replica = 0;
       replica < replica_pages && replica < ranked.size() && remaining >= 2;
       ++replica) {
    const uint32_t center = ranked[replica];
    std::unordered_map<uint32_t, uint32_t> neighbor_count;
    for (uint32_t edge : node_edges[center]) {
      for (uint32_t vertex : edges[edge]) {
        if (primary[vertex] != primary[center]) {
          ++neighbor_count[vertex];
        }
      }
    }
    std::vector<std::pair<uint32_t, uint32_t>> neighbors;
    neighbors.reserve(neighbor_count.size());
    for (const auto& item : neighbor_count) {
      neighbors.push_back({item.second, item.first});
    }
    std::sort(neighbors.begin(), neighbors.end(), [](const auto& a, const auto& b) {
      return a.first > b.first || (a.first == b.first && a.second < b.second);
    });
    std::vector<uint32_t> chosen{center};
    for (const auto& neighbor : neighbors) {
      if (chosen.size() == std::min<uint64_t>(cap, remaining)) {
        break;
      }
      chosen.push_back(neighbor.second);
    }
    if (chosen.size() < 2) {
      continue;
    }
    const uint32_t page = page_count++;
    for (uint32_t vertex : chosen) {
      pages[vertex].push_back(page);
    }
    remaining -= chosen.size();
  }
  return Finish(Strategy::kMaxEmbed, partitioned.page_size, partitioned.row_size,
                RowsFromParts(keys, pages, partitioned.row_size));
}

Artifact Shrink(const Artifact& replicated, double rep_ratio) {
  if (!std::isfinite(rep_ratio) || rep_ratio < 0.0) {
    Fail("shrink ratio must be finite and non-negative");
  }
  const long double requested =
      static_cast<long double>(replicated.keys.size()) *
      (1.0L + static_cast<long double>(rep_ratio));
  if (requested > std::numeric_limits<uint64_t>::max()) {
    Fail("shrink ratio is too large");
  }
  const uint64_t target = std::max<uint64_t>(
      replicated.keys.size(),
      static_cast<uint64_t>(std::ceil(requested)));
  std::vector<char> primary(replicated.pages.size(), 0);
  uint64_t copies = 0;
  for (const auto& row : replicated.copies) {
    copies += row.size();
    if (!row.empty() && row.front().page_id < primary.size()) {
      primary[row.front().page_id] = 1;
    }
  }
  std::vector<char> drop(replicated.pages.size(), 0);
  for (uint32_t page = static_cast<uint32_t>(drop.size());
       page > 0 && copies > target; --page) {
    if (primary[page - 1]) {
      continue;
    }
    copies -= replicated.pages[page - 1].size();
    drop[page - 1] = 1;
  }
  std::vector<uint32_t> remap(drop.size(), 0);
  uint32_t kept = 0;
  for (size_t page = 0; page < drop.size(); ++page) {
    if (!drop[page]) {
      remap[page] = kept++;
    }
  }
  Rows rows;
  rows.keys = replicated.keys;
  rows.copies.resize(replicated.copies.size());
  for (size_t i = 0; i < replicated.copies.size(); ++i) {
    for (const Location& copy : replicated.copies[i]) {
      if (!drop[copy.page_id]) {
        rows.copies[i].push_back(Location{remap[copy.page_id], copy.byte_offset});
      }
    }
  }
  return Finish(replicated.strategy, replicated.page_size, replicated.row_size,
                std::move(rows));
}

Artifact Normalize(Artifact artifact) {
  Rows rows;
  rows.keys   = std::move(artifact.keys);
  rows.copies = std::move(artifact.copies);
  return Finish(artifact.strategy, artifact.page_size, artifact.row_size,
                std::move(rows));
}

void SaveArtifact(const Artifact& artifact, const std::string& path) {
  CheckArtifact(artifact, artifact.strategy, artifact.row_size);
  std::vector<unsigned char> bytes;
  AppendPod(&bytes, kMagic, sizeof(kMagic));
  Append<uint32_t>(&bytes, kVersion);
  Append<uint32_t>(&bytes, static_cast<uint32_t>(artifact.strategy));
  Append<uint32_t>(&bytes, artifact.page_size);
  Append<uint32_t>(&bytes, artifact.row_size);
  Append<uint64_t>(&bytes, artifact.keys.size());
  Append<uint64_t>(&bytes, artifact.pages.size());
  uint64_t copy_count = 0;
  for (const auto& row : artifact.copies) {
    copy_count += row.size();
  }
  Append<uint64_t>(&bytes, copy_count);
  for (size_t i = 0; i < artifact.keys.size(); ++i) {
    Append<uint64_t>(&bytes, artifact.keys[i]);
    Append<uint32_t>(&bytes, static_cast<uint32_t>(artifact.copies[i].size()));
    for (const Location& copy : artifact.copies[i]) {
      Append<uint64_t>(&bytes, copy.page_id);
      Append<uint32_t>(&bytes, copy.byte_offset);
    }
  }
  for (const auto& page : artifact.pages) {
    Append<uint32_t>(&bytes, static_cast<uint32_t>(page.size()));
    for (const auto& item : page) {
      Append<uint64_t>(&bytes, item.first);
      Append<uint32_t>(&bytes, item.second);
    }
  }
  Append<uint64_t>(&bytes, Fnv(bytes.data(), bytes.size()));
  std::ofstream output(path, std::ios::binary | std::ios::trunc);
  if (!output.write(reinterpret_cast<const char*>(bytes.data()),
                    static_cast<std::streamsize>(bytes.size()))) {
    Fail("cannot write " + path);
  }
}

Artifact LoadArtifact(const std::string& path) {
  std::vector<unsigned char> bytes;
  ReadAll(path, &bytes);
  if (bytes.size() < sizeof(kMagic) + sizeof(uint64_t)) {
    Fail("truncated SSD layout artifact");
  }
  const uint64_t stored = [&] {
    uint64_t value = 0;
    std::memcpy(&value, bytes.data() + bytes.size() - sizeof(value), sizeof(value));
    return value;
  }();
  if (Fnv(bytes.data(), bytes.size() - sizeof(stored)) != stored) {
    Fail("SSD layout checksum mismatch");
  }
  size_t offset = 0;
  char magic[8];
  std::memcpy(magic, bytes.data(), sizeof(magic));
  offset += sizeof(magic);
  if (std::memcmp(magic, kMagic, sizeof(magic)) != 0 ||
      ReadAt<uint32_t>(bytes, &offset) != kVersion) {
    Fail("unsupported SSD layout artifact");
  }
  Artifact artifact;
  artifact.checksum   = stored;
  artifact.strategy   = static_cast<Strategy>(ReadAt<uint32_t>(bytes, &offset));
  artifact.page_size  = ReadAt<uint32_t>(bytes, &offset);
  artifact.row_size   = ReadAt<uint32_t>(bytes, &offset);
  const uint64_t key_count = ReadAt<uint64_t>(bytes, &offset);
  const uint64_t page_count = ReadAt<uint64_t>(bytes, &offset);
  const uint64_t copy_count = ReadAt<uint64_t>(bytes, &offset);
  artifact.keys.reserve(key_count);
  artifact.copies.reserve(key_count);
  uint64_t seen_copies = 0;
  for (uint64_t i = 0; i < key_count; ++i) {
    artifact.keys.push_back(ReadAt<uint64_t>(bytes, &offset));
    const uint32_t count = ReadAt<uint32_t>(bytes, &offset);
    std::vector<Location> copies(count);
    for (uint32_t copy = 0; copy < count; ++copy) {
      copies[copy].page_id     = ReadAt<uint64_t>(bytes, &offset);
      copies[copy].byte_offset = ReadAt<uint32_t>(bytes, &offset);
    }
    seen_copies += count;
    artifact.copies.push_back(std::move(copies));
  }
  artifact.pages.resize(page_count);
  for (uint64_t page = 0; page < page_count; ++page) {
    const uint32_t count = ReadAt<uint32_t>(bytes, &offset);
    artifact.pages[page].resize(count);
    for (uint32_t item = 0; item < count; ++item) {
      artifact.pages[page][item].first  = ReadAt<uint64_t>(bytes, &offset);
      artifact.pages[page][item].second = ReadAt<uint32_t>(bytes, &offset);
    }
  }
  if (seen_copies != copy_count || offset + sizeof(uint64_t) != bytes.size()) {
    Fail("SSD layout artifact has trailing or mismatched records");
  }
  Rows rows;
  rows.keys   = artifact.keys;
  rows.copies = artifact.copies;
  Artifact checked = Finish(artifact.strategy, artifact.page_size, artifact.row_size,
                            std::move(rows));
  if (checked.pages != artifact.pages) {
    Fail("SSD layout page index does not match key placement");
  }
  checked.checksum = stored;
  return checked;
}

void CheckArtifact(const Artifact& artifact,
                   Strategy expected,
                   uint32_t expected_row_size) {
  if (artifact.strategy != expected || artifact.row_size != expected_row_size) {
    Fail("SSD layout strategy or row size mismatch");
  }
  Rows rows;
  rows.keys   = artifact.keys;
  rows.copies = artifact.copies;
  const Artifact checked = Finish(expected, artifact.page_size, expected_row_size,
                                  std::move(rows));
  if (checked.pages != artifact.pages) {
    Fail("SSD layout page index does not match key placement");
  }
}

} // namespace ssd_layout
