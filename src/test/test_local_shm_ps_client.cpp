#include <gtest/gtest.h>

#include <cstdlib>
#include <chrono>
#include <limits>
#include <thread>
#include <vector>

#include "base/json.h"
#include "ps/local_shm/local_shm_client.h"
#include "ps/local_shm/local_shm_queue.h"
#define private public
#include "ps/local_shm/local_shm_server.h"
#undef private

namespace recstore {
namespace {

json MakeLocalShmConfig(const std::string& region_name,
                        uint32_t ready_queue_count       = 1,
                        uint32_t ready_queue_burst_limit = 8) {
  return {
      {"cache_ps",
       {{"num_threads", 1},
        {"ps_type", "LOCAL_SHM"},
        {"base_kv_config",
         {{"path", "/tmp/recstore_local_shm_test"},
          {"index_type", "DRAM"},
          {"value_type", "DRAM"},
          {"capacity", 1024},
          {"value_size", 16}}}}},
      {"local_shm",
       {{"region_name", region_name},
        {"slot_count", 8},
        {"ready_queue_count", ready_queue_count},
        {"ready_queue_burst_limit", ready_queue_burst_limit},
        {"slot_buffer_bytes", 1 << 20},
        {"client_timeout_ms", 1000}}},
  };
}

TEST(LocalShmPSClientTest, FactoryClientTypeCanBeConstructed) {
  const auto config = MakeLocalShmConfig(
      "recstore_local_shm_ps_client_factory_" + std::to_string(::getpid()));
  LocalShmParameterServer server;
  server.Init(config);

  std::thread server_thread([&]() { server.Run(); });

  std::this_thread::sleep_for(std::chrono::milliseconds(10));

  LocalShmPSClient client(config["local_shm"]);
  EXPECT_EQ(client.InitEmbeddingTable("table_a", {128, 4}), 0);

  server.Stop();
  server_thread.join();
}

TEST(LocalShmPSClientTest, PutGetAndUpdateFlatRoundTrip) {
  const auto config = MakeLocalShmConfig(
      "recstore_local_shm_ps_client_rw_" + std::to_string(::getpid()));
  LocalShmParameterServer server;
  server.Init(config);

  std::thread server_thread([&]() { server.Run(); });
  std::this_thread::sleep_for(std::chrono::milliseconds(10));

  LocalShmPSClient client(config["local_shm"]);
  ASSERT_EQ(client.InitEmbeddingTable("table_b", {128, 4}), 0);

  std::vector<uint64_t> keys             = {1, 5};
  std::vector<std::vector<float>> values = {
      {1.0f, 2.0f, 3.0f, 4.0f}, {5.0f, 6.0f, 7.0f, 8.0f}};
  base::ConstArray<uint64_t> key_array(keys);
  ASSERT_EQ(client.PutParameter(key_array, values), 0);

  std::vector<float> readback(8, 0.0f);
  ASSERT_EQ(client.GetParameter(key_array, readback.data()), 0);
  EXPECT_EQ(readback[0], 1.0f);
  EXPECT_EQ(readback[1], 2.0f);
  EXPECT_EQ(readback[4], 5.0f);
  EXPECT_EQ(readback[7], 8.0f);

  std::vector<float> grads = {1.0f, 1.0f, 1.0f, 1.0f, 2.0f, 2.0f, 2.0f, 2.0f};
  ASSERT_EQ(
      client.UpdateParameterFlat("table_b", key_array, grads.data(), 2, 4), 0);
  ASSERT_EQ(client.GetParameter(key_array, readback.data()), 0);

  server.Stop();
  server_thread.join();
}

TEST(LocalShmPSClientTest, GetParameterFlatRoundTripUsesFixedEmbeddingDim) {
  const auto config = MakeLocalShmConfig(
      "recstore_local_shm_ps_client_flat_get_" + std::to_string(::getpid()));
  LocalShmParameterServer server;
  server.Init(config);

  std::thread server_thread([&]() { server.Run(); });
  std::this_thread::sleep_for(std::chrono::milliseconds(10));

  LocalShmPSClient client(config["local_shm"]);
  ASSERT_EQ(client.InitEmbeddingTable("table_flat", {128, 4}), 0);

  std::vector<uint64_t> keys             = {2, 6};
  std::vector<std::vector<float>> values = {
      {2.0f, 3.0f, 4.0f, 5.0f}, {6.0f, 7.0f, 8.0f, 9.0f}};
  base::ConstArray<uint64_t> key_array(keys);
  ASSERT_EQ(client.PutParameter(key_array, values), 0);

  std::vector<float> readback(8, 0.0f);
  ASSERT_EQ(client.GetParameterFlat(key_array, readback.data(), 2, 4), 0);
  EXPECT_EQ(readback[0], 2.0f);
  EXPECT_EQ(readback[1], 3.0f);
  EXPECT_EQ(readback[4], 6.0f);
  EXPECT_EQ(readback[7], 9.0f);

  server.Stop();
  server_thread.join();
}

TEST(LocalShmPSClientTest, SubmitWaitAndReleaseGetParameterFlatExposesSlotPayload) {
  const auto config = MakeLocalShmConfig(
      "recstore_local_shm_ps_client_flat_handle_" + std::to_string(::getpid()));
  LocalShmParameterServer server;
  server.Init(config);

  std::thread server_thread([&]() { server.Run(); });
  std::this_thread::sleep_for(std::chrono::milliseconds(10));

  LocalShmPSClient client(config["local_shm"]);
  ASSERT_EQ(client.InitEmbeddingTable("table_slot", {128, 4}), 0);

  std::vector<uint64_t> keys             = {9, 13};
  std::vector<std::vector<float>> values = {
      {9.0f, 10.0f, 11.0f, 12.0f}, {13.0f, 14.0f, 15.0f, 16.0f}};
  base::ConstArray<uint64_t> key_array(keys);
  ASSERT_EQ(client.PutParameter(key_array, values), 0);

  LocalShmFlatGetHandle handle;
  ASSERT_EQ(client.SubmitGetParameterFlat(key_array, 2, 4, &handle), 0);
  ASSERT_NE(handle.slot_id, std::numeric_limits<uint32_t>::max());
  ASSERT_EQ(client.WaitGetParameterFlat(&handle), 0);
  ASSERT_NE(handle.values, nullptr);
  ASSERT_EQ(handle.num_rows, 2);
  ASSERT_EQ(handle.embedding_dim, 4);
  ASSERT_EQ(handle.output_bytes, sizeof(float) * 8);

  EXPECT_FLOAT_EQ(handle.values[0], 9.0f);
  EXPECT_FLOAT_EQ(handle.values[1], 10.0f);
  EXPECT_FLOAT_EQ(handle.values[4], 13.0f);
  EXPECT_FLOAT_EQ(handle.values[7], 16.0f);

  client.ReleaseGetParameterFlat(&handle);
  EXPECT_EQ(handle.values, nullptr);
  EXPECT_EQ(handle.num_rows, 0);
  EXPECT_EQ(handle.embedding_dim, 0);
  EXPECT_EQ(handle.output_bytes, 0);

  server.Stop();
  server_thread.join();
}

TEST(LocalShmPSClientTest, GetParameterFlatRejectsMismatchedEmbeddingDim) {
  const auto config = MakeLocalShmConfig(
      "recstore_local_shm_ps_client_flat_dim_mismatch_" +
      std::to_string(::getpid()));
  LocalShmParameterServer server;
  server.Init(config);

  std::thread server_thread([&]() { server.Run(); });
  std::this_thread::sleep_for(std::chrono::milliseconds(10));

  LocalShmPSClient client(config["local_shm"]);
  ASSERT_EQ(client.InitEmbeddingTable("table_mismatch", {128, 4}), 0);

  std::vector<uint64_t> keys             = {7};
  std::vector<std::vector<float>> values = {{7.0f, 8.0f, 9.0f, 10.0f}};
  base::ConstArray<uint64_t> key_array(keys);
  ASSERT_EQ(client.PutParameter(key_array, values), 0);

  std::vector<float> readback(3, 0.0f);
  EXPECT_EQ(client.GetParameterFlat(key_array, readback.data(), 1, 3), -1);

  server.Stop();
  server_thread.join();
}

TEST(LocalShmPSClientTest, UpdateParameterFlatRejectsMismatchedEmbeddingDim) {
  const auto config = MakeLocalShmConfig(
      "recstore_local_shm_ps_client_update_dim_mismatch_" +
      std::to_string(::getpid()));
  LocalShmParameterServer server;
  server.Init(config);

  std::thread server_thread([&]() { server.Run(); });
  std::this_thread::sleep_for(std::chrono::milliseconds(10));

  LocalShmPSClient client(config["local_shm"]);
  ASSERT_EQ(client.InitEmbeddingTable("table_update_mismatch", {128, 4}), 0);

  std::vector<uint64_t> keys             = {11};
  std::vector<std::vector<float>> values = {{1.0f, 2.0f, 3.0f, 4.0f}};
  base::ConstArray<uint64_t> key_array(keys);
  ASSERT_EQ(client.PutParameter(key_array, values), 0);

  std::vector<float> grads = {0.1f, 0.2f, 0.3f};
  EXPECT_EQ(
      client.UpdateParameterFlat(
          "table_update_mismatch", key_array, grads.data(), 1, 3),
      -1);

  server.Stop();
  server_thread.join();
}

TEST(LocalShmPSClientTest, MultiClientUsesIndependentReadyQueues) {
  const auto config = MakeLocalShmConfig(
      "recstore_local_shm_ps_client_multi_" + std::to_string(::getpid()), 2);
  LocalShmParameterServer server;
  server.Init(config);

  std::thread server_thread([&]() { server.Run(); });
  std::this_thread::sleep_for(std::chrono::milliseconds(10));

  json client0_config                 = config["local_shm"];
  json client1_config                 = config["local_shm"];
  client0_config["ready_queue_index"] = 0;
  client1_config["ready_queue_index"] = 1;

  LocalShmPSClient client0(client0_config);
  LocalShmPSClient client1(client1_config);
  ASSERT_EQ(client0.InitEmbeddingTable("table_c", {128, 4}), 0);

  std::thread worker0([&]() {
    std::vector<uint64_t> keys             = {1, 3};
    std::vector<std::vector<float>> values = {
        {1.0f, 1.0f, 1.0f, 1.0f}, {3.0f, 3.0f, 3.0f, 3.0f}};
    base::ConstArray<uint64_t> key_array(keys);
    EXPECT_EQ(client0.PutParameter(key_array, values), 0);
    std::vector<float> readback(8, 0.0f);
    EXPECT_EQ(client0.GetParameter(key_array, readback.data()), 0);
    EXPECT_FLOAT_EQ(readback[0], 1.0f);
    EXPECT_FLOAT_EQ(readback[4], 3.0f);
  });

  std::thread worker1([&]() {
    std::vector<uint64_t> keys             = {2, 4};
    std::vector<std::vector<float>> values = {
        {2.0f, 2.0f, 2.0f, 2.0f}, {4.0f, 4.0f, 4.0f, 4.0f}};
    base::ConstArray<uint64_t> key_array(keys);
    EXPECT_EQ(client1.PutParameter(key_array, values), 0);
    std::vector<float> readback(8, 0.0f);
    EXPECT_EQ(client1.GetParameter(key_array, readback.data()), 0);
    EXPECT_FLOAT_EQ(readback[0], 2.0f);
    EXPECT_FLOAT_EQ(readback[4], 4.0f);
  });

  worker0.join();
  worker1.join();

  server.Stop();
  server_thread.join();
}

TEST(LocalShmPSClientTest, LocalRankEnvironmentSelectsReadyQueue) {
  auto config = MakeLocalShmConfig(
      "recstore_local_shm_ps_client_env_" + std::to_string(::getpid()), 2);
  config["local_shm"]["client_timeout_ms"] = 100;
  LocalShmRegion region;
  ASSERT_TRUE(region.Create(
      config["local_shm"]["region_name"].get<std::string>(), 8, 1 << 20, 2));

  ASSERT_EQ(::setenv("LOCAL_RANK", "1", 1), 0);
  int init_ret = 0;
  std::thread worker([&]() {
    LocalShmPSClient client(config["local_shm"]);
    init_ret = client.InitEmbeddingTable("table_env", {128, 4});
  });

  std::this_thread::sleep_for(std::chrono::milliseconds(20));
  uint32_t slot_id = 0;
  EXPECT_FALSE(LocalShmQueueDequeue(
      region.ready_queue_header(0), region.ready_queue_cells(0), &slot_id));
  EXPECT_TRUE(LocalShmQueueDequeue(
      region.ready_queue_header(1), region.ready_queue_cells(1), &slot_id));

  worker.join();
  EXPECT_EQ(init_ret, -1);
  ::unsetenv("LOCAL_RANK");
}

TEST(LocalShmPSClientTest, ServerDrainReadyQueueHonorsBurstLimit) {
  const auto config = MakeLocalShmConfig(
      "recstore_local_shm_server_burst_" + std::to_string(::getpid()), 2, 2);
  LocalShmRegion region;
  ASSERT_TRUE(region.Create(
      config["local_shm"]["region_name"].get<std::string>(), 8, 1 << 20, 2));

  for (uint32_t slot_id = 0; slot_id < 3; ++slot_id) {
    auto* header   = region.slot_header(slot_id);
    header->opcode = static_cast<uint32_t>(LocalOpcode::kInvalid);
    header->state.store(static_cast<uint32_t>(LocalSlotState::kReady),
                        std::memory_order_release);
    ASSERT_TRUE(LocalShmQueueEnqueue(
        region.ready_queue_header(0), region.ready_queue_cells(0), slot_id));
  }

  LocalShmStoreRuntime runtime(&region, nullptr, 2);
  uint32_t processed = 0;
  EXPECT_TRUE(runtime.DrainReadyQueue(0, &processed));
  EXPECT_EQ(processed, 2U);

  EXPECT_EQ(region.slot_header(0)->state.load(std::memory_order_acquire),
            static_cast<uint32_t>(LocalSlotState::kError));
  EXPECT_EQ(region.slot_header(1)->state.load(std::memory_order_acquire),
            static_cast<uint32_t>(LocalSlotState::kError));
  EXPECT_EQ(region.slot_header(2)->state.load(std::memory_order_acquire),
            static_cast<uint32_t>(LocalSlotState::kReady));

  uint32_t remaining_slot_id = 0;
  EXPECT_TRUE(LocalShmQueueDequeue(
      region.ready_queue_header(0),
      region.ready_queue_cells(0),
      &remaining_slot_id));
  EXPECT_EQ(remaining_slot_id, 2U);
}

} // namespace
} // namespace recstore
