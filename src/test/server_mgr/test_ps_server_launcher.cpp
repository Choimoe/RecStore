#include "ps_server_launcher.h"

#include <fstream>

#include <gtest/gtest.h>

#include "base/json.h"

namespace recstore::test {
namespace {

TEST(PSServerLauncherUnitTest, ParseReadyShardWorks) {
  auto shard = PSServerLauncher::ParseReadyShard(
      "Server shard 1 listening on 127.0.0.1:15001");
  ASSERT_TRUE(shard.has_value());
  EXPECT_EQ(*shard, 1);

  auto single =
      PSServerLauncher::ParseReadyShard("Server listening on 127.0.0.1:15000");
  ASSERT_TRUE(single.has_value());
  EXPECT_EQ(*single, 0);

  auto invalid = PSServerLauncher::ParseReadyShard("random output");
  EXPECT_FALSE(invalid.has_value());
}

TEST(PSServerLauncherUnitTest, FormatsRecentLogsForError) {
  std::vector<std::string> logs = {
      "Starting PS Server: /tmp/ps_server",
      "[STDERR] no Factory instance",
      "[STDERR] aborting"};

  auto message =
      PSServerLauncher::FormatRecentLogsForErrorForTest(logs, 2);

  EXPECT_NE(message.find("Recent ps_server output:"), std::string::npos);
  EXPECT_EQ(message.find("Starting PS Server"), std::string::npos);
  EXPECT_NE(message.find("[STDERR] no Factory instance"), std::string::npos);
  EXPECT_NE(message.find("[STDERR] aborting"), std::string::npos);
}

TEST(PSServerLauncherUnitTest, ExtractPortsFromConfigSupportsCachePS) {
  const std::filesystem::path config_path =
      std::filesystem::temp_directory_path() /
      "ps_server_launcher_cache_ps_config.json";

  {
    std::ofstream out(config_path);
    out << R"({
      "cache_ps": {
        "servers": [
          {"host": "127.0.0.1", "port": 15123},
          {"host": "127.0.0.1", "port": 15124},
          {"host": "127.0.0.1", "port": 15123}
        ]
      }
    })";
  }

  auto ports = PSServerLauncher::ExtractPortsFromConfig(config_path);
  ASSERT_EQ(ports.size(), 2);
  EXPECT_EQ(ports[0], 15123);
  EXPECT_EQ(ports[1], 15124);

  std::error_code ec;
  std::filesystem::remove(config_path, ec);
}

TEST(PSServerLauncherUnitTest, ExtractPortsFromConfigFallsBack) {
  auto ports =
      PSServerLauncher::ExtractPortsFromConfig("/path/not/exist/config.json");
  EXPECT_EQ(ports.size(), 4);
  EXPECT_EQ(ports[0], 15000);
}

TEST(PSServerLauncherUnitTest, EvaluateDecisionForNoServerEnv) {
  setenv("NO_PS_SERVER", "1", 1);

  LauncherOptions options;
  options.config_path = "/path/not/exist/config.json";
  auto decision       = PSServerLauncher::EvaluateLaunchDecision(options);

  EXPECT_FALSE(decision.should_start);
  EXPECT_FALSE(decision.should_fail);
  EXPECT_EQ(decision.reason, "NO_PS_SERVER");

  unsetenv("NO_PS_SERVER");
}

TEST(PSServerLauncherUnitTest, LoadOptionsRespectsEnvOverride) {
  setenv("PS_SERVER_PATH", "/tmp/ps_server_custom", 1);
  setenv("PS_TIMEOUT", "17", 1);
  setenv("PS_NUM_SHARDS", "3", 1);
  setenv("PS_LOCAL_SHARD_ID", "1", 1);

  auto options = PSServerLauncher::LoadOptionsFromEnvironment();
  EXPECT_EQ(options.server_path,
            std::filesystem::path("/tmp/ps_server_custom"));
  EXPECT_EQ(options.startup_timeout_sec, 17);
  EXPECT_EQ(options.num_shards, 1);
  ASSERT_TRUE(options.local_shard_id.has_value());
  EXPECT_EQ(*options.local_shard_id, 1);

  unsetenv("PS_SERVER_PATH");
  unsetenv("PS_TIMEOUT");
  unsetenv("PS_NUM_SHARDS");
  unsetenv("PS_LOCAL_SHARD_ID");
}

TEST(PSServerLauncherUnitTest, PrepareConfigForLaunchAllowsLocalShardOverride) {
  const std::filesystem::path config_path =
      std::filesystem::temp_directory_path() /
      "ps_server_launcher_local_shard_config.json";

  {
    std::ofstream out(config_path);
    out << R"({
      "cache_ps": {
        "num_shards": 2,
        "servers": [
          {"host": "127.0.0.1", "port": 15123, "shard": 0},
          {"host": "127.0.0.1", "port": 15124, "shard": 1}
        ]
      },
      "distributed_client": {
        "num_shards": 2,
        "servers": [
          {"host": "127.0.0.1", "port": 15123, "shard": 0},
          {"host": "127.0.0.1", "port": 15124, "shard": 1}
        ]
      }
    })";
  }

  LauncherOptions options;
  options.config_path    = config_path;
  options.log_dir        = std::filesystem::temp_directory_path();
  options.local_shard_id = 1;
  options.num_shards     = 1;

  PSServerLauncher launcher(options);
  auto generated_config = launcher.PrepareConfigForLaunchForTest();
  ASSERT_FALSE(generated_config.empty());
  ASSERT_TRUE(std::filesystem::exists(generated_config));

  std::ifstream in(generated_config);
  ASSERT_TRUE(in.good());
  nlohmann::json cfg;
  in >> cfg;
  EXPECT_EQ(cfg["cache_ps"]["num_shards"].get<int>(), 2);
  ASSERT_EQ(cfg["cache_ps"]["servers"].size(), 2);

  launcher.CleanupLaunchConfigForTest();

  std::error_code ec;
  std::filesystem::remove(config_path, ec);
}

} // namespace
} // namespace recstore::test
