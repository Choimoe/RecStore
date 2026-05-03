#include <fstream>

#include <gflags/gflags.h>

#include "base/init.h"
#include "base/json.h"
#include "base/log.h"
#include "ps/local_shm/local_shm_server.h"

DEFINE_string(config_path,
              "./recstore_config.json",
              "Path to the recstore config for local_shm_server");

int main(int argc, char** argv) {
  base::Init(&argc, &argv);

  std::ifstream config_file(FLAGS_config_path);
  if (!config_file.good()) {
    LOG(ERROR) << "Failed to open config file: " << FLAGS_config_path;
    return 1;
  }

  json config;
  config_file >> config;

  recstore::LocalShmParameterServer server;
  server.Init(config);
  server.Run();
  return 0;
}
