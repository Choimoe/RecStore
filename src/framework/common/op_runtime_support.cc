#include "framework/common/op_runtime_support.h"

#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <mutex>
#include <stdexcept>

#include <glog/logging.h>

#include "base/config.h"
#include "framework/common/ps_client_config_adapter.h"
#include "ps/client_factory.h"
#include "ps/rdma/rdma_ps_client_adapter.h"

namespace recstore {

std::string BackendNameFromConfig(const json& config) {
  switch (ResolveFrameworkPSClientType(config)) {
  case PSClientType::kGrpc:
    return "grpc";
  case PSClientType::kBrpc:
    return "brpc";
  case PSClientType::kRdma:
    return "rdma";
  case PSClientType::kLocalShm:
    return "local_shm";
  }

  return "unknown";
}

json load_config_from_file(const std::string& config_path) {
  std::ifstream file(config_path);
  if (!file.is_open()) {
    throw std::runtime_error("Cannot open config file: " + config_path);
  }

  std::string content(
      (std::istreambuf_iterator<char>(file)), std::istreambuf_iterator<char>());
  file.close();

  try {
    return json::parse(content);
  } catch (const json::exception& e) {
    throw std::runtime_error(
        "Failed to parse config file: " + std::string(e.what()));
  }
}

std::unique_ptr<BasePSClient> create_ps_client_from_config(const json& config) {
  return CreatePSClient(ResolvePSClientOptionsFromFrameworkConfig(config));
}

json GetGlobalConfig() {
  try {
    auto current_path = std::filesystem::current_path();
    std::cerr << "[Config] Current working directory: " << current_path.string()
              << std::endl;

    std::filesystem::path config_path;

    if (const char* env_config = std::getenv("RECSTORE_CONFIG");
        env_config != nullptr && env_config[0] != '\0') {
      config_path = env_config;
      if (std::filesystem::exists(config_path)) {
        std::cerr << "[Config] Using config file from RECSTORE_CONFIG: "
                  << config_path.string() << std::endl;
      } else {
        throw std::runtime_error("RECSTORE_CONFIG points to a missing file: " +
                                 config_path.string());
      }
    } else {
      config_path = base::ResolveRecStoreConfigPath();
      std::cerr << "[Config] Found config file at: " << config_path.string()
                << std::endl;
    }

    std::ifstream test_file(config_path);
    if (!test_file.good()) {
      throw std::runtime_error(
          "Config file not found: " + config_path.string() +
          ". Please ensure recstore_config.json exists "
          "in the project root directory.");
    }
    test_file.close();

    return load_config_from_file(config_path);
  } catch (const std::exception& e) {
    std::cerr << "Failed to load config: " << e.what() << std::endl;
    return json::object();
  }
}

void ConfigureLogging(bool initialize_google_logging) {
  static std::once_flag log_init_flag;
  std::call_once(log_init_flag, [initialize_google_logging]() {
    std::cerr << "[Debug] ConfigureLogging called. Setting flags." << std::endl;
    FLAGS_log_dir         = "/tmp";
    FLAGS_alsologtostderr = false;
    FLAGS_logtostderr     = false;
    FLAGS_stderrthreshold = google::ERROR;

    google::SetLogDestination(google::INFO, "/tmp/recstore_op_layer_INFO_");
    google::SetLogDestination(
        google::WARNING, "/tmp/recstore_op_layer_WARNING_");
    google::SetLogDestination(google::ERROR, "/tmp/recstore_op_layer_ERROR_");

    if (initialize_google_logging) {
      google::InitGoogleLogging("recstore_op_layer");
    }
  });
}

} // namespace recstore
