#include "storage/ssd_layout/ssd_layout.h"

#include <cstdlib>
#include <iostream>
#include <string>

namespace {

void Usage() {
  std::cerr << "usage:\n"
            << "  recstore_ssd_layout partition --query PATH --output PATH\n"
            << "      --row-size N [--page-size 4096] [--keys PATH]\n"
            << "      [--binary]\n"
            << "  recstore_ssd_layout replicate --query PATH --mapping PATH\n"
            << "      --output PATH --rep-ratio F [--keys PATH] [--binary]\n"
            << "  recstore_ssd_layout shrink --mapping PATH --output PATH\n"
            << "      --rep-ratio F\n"
            << "  recstore_ssd_layout inspect --mapping PATH\n";
}

std::string Need(int argc, char** argv, int* i) {
  if (*i + 1 >= argc) {
    Usage();
    std::exit(2);
  }
  return argv[++(*i)];
}

} // namespace

int main(int argc, char** argv) {
  try {
    if (argc < 2) {
      Usage();
      return 2;
    }
    const std::string command = argv[1];
    std::string query;
    std::string mapping;
    std::string output;
    std::string keys_path;
    uint32_t row_size  = 0;
    uint32_t page_size = 4096;
    double rep_ratio   = 0;
    bool binary        = false;
    for (int i = 2; i < argc; ++i) {
      const std::string flag = argv[i];
      if (flag == "--binary") {
        binary = true;
      } else if (flag == "--query") {
        query = Need(argc, argv, &i);
      } else if (flag == "--mapping") {
        mapping = Need(argc, argv, &i);
      } else if (flag == "--output") {
        output = Need(argc, argv, &i);
      } else if (flag == "--keys") {
        keys_path = Need(argc, argv, &i);
      } else if (flag == "--row-size") {
        row_size = static_cast<uint32_t>(std::stoul(Need(argc, argv, &i)));
      } else if (flag == "--page-size") {
        page_size = static_cast<uint32_t>(std::stoul(Need(argc, argv, &i)));
      } else if (flag == "--rep-ratio") {
        rep_ratio = std::stod(Need(argc, argv, &i));
      } else {
        Usage();
        return 2;
      }
    }

    if (command == "inspect") {
      const ssd_layout::Artifact artifact = ssd_layout::LoadArtifact(mapping);
      uint64_t copies                     = 0;
      for (const auto& row : artifact.copies) {
        copies += row.size();
      }
      std::cout << "strategy=" << ssd_layout::StrategyName(artifact.strategy)
                << " page_size=" << artifact.page_size << " row_size="
                << artifact.row_size << " keys=" << artifact.keys.size()
                << " pages=" << artifact.pages.size() << " copies=" << copies
                << " checksum=" << artifact.checksum << "\n";
      return 0;
    }
    if (output.empty()) {
      Usage();
      return 2;
    }
    if (command == "partition") {
      const ssd_layout::Hypergraph graph =
          ssd_layout::ReadQueryLog(query, binary);
      const auto keys = ssd_layout::ReadKeyMap(keys_path, graph.vertex_count);
      ssd_layout::SaveArtifact(
          ssd_layout::Partition(graph, keys, page_size, row_size), output);
      return 0;
    }
    if (command == "replicate") {
      const ssd_layout::Hypergraph graph =
          ssd_layout::ReadQueryLog(query, binary);
      const auto keys = ssd_layout::ReadKeyMap(keys_path, graph.vertex_count);
      const ssd_layout::Artifact base = ssd_layout::LoadArtifact(mapping);
      auto replicated = ssd_layout::Replicate(base, graph, keys, rep_ratio);
      ssd_layout::SaveArtifact(replicated, output);
      return 0;
    }
    if (command == "shrink") {
      ssd_layout::SaveArtifact(
          ssd_layout::Shrink(ssd_layout::LoadArtifact(mapping), rep_ratio),
          output);
      return 0;
    }
    Usage();
    return 2;
  } catch (const std::exception& error) {
    std::cerr << "recstore_ssd_layout: " << error.what() << "\n";
    return 1;
  }
}
