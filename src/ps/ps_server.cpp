#include <csignal>
#include <iostream>
#include <fstream>
#include <string>
#include <algorithm>
#include <cstdlib>

#include <gflags/gflags.h>

#include "base/base.h"
#include "base/factory.h"
#include "base/init.h"
#include "base/json.h"
#include "ps/base/base_ps_server.h"
#include "src/base/config.h"

#ifdef ENABLE_PERF_REPORT
#  include <chrono>
#  include "base/report/report_client.h"
#endif

#ifdef ENABLE_GPERF_PROFILING
#  include <gperftools/profiler.h>
#endif

DECLARE_string(config_path);
DECLARE_string(brpc_config_path);

using recstore::BaseParameterServer;

#ifdef ENABLE_GPERF_PROFILING
void StopProfilerAndExit(int signum) {
  LOG(INFO) << "Caught signal " << signum << ", stopping gperf profiler.";
  ProfilerStop();
  std::exit(signum);
}
#endif

static inline std::string ToUpper(std::string s) {
  std::transform(s.begin(), s.end(), s.begin(), [](unsigned char c) {
    return std::toupper(c);
  });
  return s;
}

int main(int argc, char** argv) {
  base::Init(&argc, &argv);

#ifdef ENABLE_GPERF_PROFILING
  std::signal(SIGINT, StopProfilerAndExit);
  std::signal(SIGTERM, StopProfilerAndExit);

  struct ProfilerGuard {
    const char* profile;
    ProfilerGuard() : profile(std::getenv("CPUPROFILE")) {
      if (profile) {
        ProfilerStart(profile);
        LOG(INFO) << "gperftools CPU profiling started, outputting to "
                  << profile;
      }
    }
    ~ProfilerGuard() {
      if (profile) {
        ProfilerStop();
        LOG(INFO) << "gperftools CPU profiling stopped.";
      }
    }
  } profiler_guard;
#endif

  std::string cfg_path = FLAGS_config_path;
  {
    if (!cfg_path.empty()) {
      std::ifstream test(cfg_path);
      if (!test.good()) {
        cfg_path.clear();
      }
    }
    if (cfg_path.empty() && !FLAGS_brpc_config_path.empty()) {
      std::ifstream test_b(FLAGS_brpc_config_path);
      if (test_b.good()) {
        cfg_path = FLAGS_brpc_config_path;
      }
    }
    if (cfg_path.empty()) {
      try {
        cfg_path = base::ResolveRecStoreConfigPath().string();
      } catch (const std::exception& e) {
        LOG(ERROR) << e.what();
        return 1;
      }
    }
  }

  std::ifstream config_file(cfg_path);
  if (!config_file.good()) {
    LOG(ERROR) << "Failed to open config file: " << cfg_path;
    return 1;
  }

  json config;
  config_file >> config;

  std::string ps_type = "GRPC";
  try {
    if (config.contains("cache_ps") && config["cache_ps"].contains("ps_type")) {
      ps_type = config["cache_ps"]["ps_type"].get<std::string>();
    }
  } catch (...) {
  }

  std::string key;
  std::string type_upper = ToUpper(ps_type);
  if (type_upper == "GRPC") {
    key = "GRPCParameterServer";
  } else if (type_upper == "BRPC") {
    key = "BRPCParameterServer";
  } else {
    LOG(ERROR) << "Unknown ps_type: " << ps_type << ", expected GRPC or BRPC";
    return 2;
  }

  LOG(INFO) << "Using ps_type: " << type_upper << " (key=" << key << ")";
  LOG(INFO) << "Parameter server config: " << config.dump(2);

  std::unique_ptr<BaseParameterServer> server(
      base::Factory<BaseParameterServer>::NewInstance(key));
  try {
#ifdef ENABLE_PERF_REPORT
    auto start_time = std::chrono::high_resolution_clock::now();
#endif
    server->Init(config);
#ifdef ENABLE_PERF_REPORT
    auto end_time = std::chrono::high_resolution_clock::now();
    auto duration =
        std::chrono::duration_cast<std::chrono::microseconds>(
            end_time - start_time)
            .count();
    report(
        "server_latency", "Init", "latency_us", static_cast<double>(duration));
#endif
    server->Run();
  } catch (const std::exception& e) {
    std::cerr << "FATAL: Uncaught exception in ps_server: " << e.what()
              << std::endl;
    LOG(FATAL) << "Uncaught exception in ps_server: " << e.what();
    return 1;
  } catch (...) {
    std::cerr << "FATAL: Unknown exception in ps_server" << std::endl;
    LOG(FATAL) << "Unknown exception in ps_server";
    return 1;
  }

  return 0;
}
