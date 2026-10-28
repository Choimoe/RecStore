#include <gtest/gtest.h>
#include "ps/base/sparse_gradient_commit.h"

using recstore::PackSparseGradientCommit;
using recstore::SparseGradientCommitCoordinator;
using recstore::SparseGradientCommitHeader;

TEST(SparseGradientCommit, ChunksEmptyRankAndRetryCommitExactlyOnce) {
  SparseGradientCommitCoordinator coordinator;
  SparseGradientCommitHeader h;
  h.session = 7;
  h.world   = 3;
  h.dim     = 2;
  h.divisor = 3;
  std::vector<uint64_t> applied_keys;
  std::vector<float> applied;
  int calls  = 0;
  auto apply = [&](const auto& keys, const auto& grads) {
    applied_keys = keys;
    applied      = grads;
    ++calls;
  };
  const uint64_t key = 9;
  const float g0[] = {3, 6}, g1[] = {6, 3};
  h.rows     = 1;
  h.chunks   = 2;
  auto first = PackSparseGradientCommit(h, &key, g0);
  EXPECT_EQ(coordinator.Exchange("table", first, apply), -1);
  EXPECT_EQ(coordinator.Exchange("table", first, apply), -1);
  EXPECT_THROW(coordinator.Exchange(
                   "table", PackSparseGradientCommit(h, &key, g1), apply),
               std::runtime_error);
  h.chunk = 1;
  h.rows  = 0;
  EXPECT_EQ(coordinator.Exchange("table", PackSparseGradientCommit(h), apply),
            -1);
  h.rank   = 1;
  h.chunk  = 0;
  h.chunks = 1;
  h.rows   = 1;
  EXPECT_EQ(coordinator.Exchange(
                "table", PackSparseGradientCommit(h, &key, g1), apply),
            -1);
  h.rank = 2;
  h.rows = 0;
  EXPECT_EQ(coordinator.Exchange("table", PackSparseGradientCommit(h), apply),
            0);
  EXPECT_EQ(calls, 1);
  EXPECT_EQ(applied_keys, std::vector<uint64_t>({9}));
  EXPECT_EQ(applied, std::vector<float>({3, 3}));
  EXPECT_EQ(coordinator.Exchange("table", first, apply), 0);
  EXPECT_EQ(calls, 1);
  h.kind = 1;
  EXPECT_EQ(coordinator.Exchange("table", PackSparseGradientCommit(h), apply),
            0);
  h.kind = 0;
  h.step = 2;
  EXPECT_THROW(
      coordinator.Exchange("table", PackSparseGradientCommit(h), apply),
      std::runtime_error);
  h.step = 1;
  for (h.rank = 0; h.rank < h.world; ++h.rank) {
    EXPECT_EQ(coordinator.Exchange("table", PackSparseGradientCommit(h), apply),
              h.rank == 2 ? 1 : 0);
  }
  EXPECT_EQ(calls, 1); // All-empty steps still advance the commit frontier.
  h.kind = 2;
  for (h.rank = 0; h.rank < h.world; ++h.rank)
    coordinator.Exchange("table", PackSparseGradientCommit(h), apply);
  h.kind    = 0;
  h.step    = 0;
  h.session = 8;
  h.rank    = 0;
  EXPECT_EQ(coordinator.Exchange("table", PackSparseGradientCommit(h), apply),
            -1);
}

TEST(SparseGradientCommit, FailedCommitCannotBeReplayed) {
  SparseGradientCommitCoordinator coordinator;
  SparseGradientCommitHeader h;
  h.session    = 4;
  h.dim        = 1;
  h.rows       = 1;
  uint64_t key = 1;
  float grad   = 2;
  auto packet  = PackSparseGradientCommit(h, &key, &grad);
  int calls    = 0;
  auto fail    = [&](const auto&, const auto&) {
    ++calls;
    throw std::runtime_error("storage failed");
  };
  EXPECT_THROW(coordinator.Exchange("table", packet, fail), std::runtime_error);
  EXPECT_THROW(coordinator.Exchange("table", packet, fail), std::runtime_error);
  EXPECT_EQ(calls, 1);
  h.rows = 0;
  h.kind = 1;
  EXPECT_THROW(coordinator.Exchange("table", PackSparseGradientCommit(h), fail),
               std::runtime_error);
  EXPECT_THROW(recstore::DecodeSparseGradientCommit(packet.substr(1)),
               std::invalid_argument);
  h.world = 0;
  EXPECT_THROW(PackSparseGradientCommit(h), std::invalid_argument);
}

TEST(SparseGradientCommit, ArrivalOrderAndRetriesPreserveAccumulationOrder) {
  SparseGradientCommitCoordinator coordinator;
  SparseGradientCommitHeader h;
  h.session               = 12;
  h.world                 = 2;
  h.dim                   = 1;
  h.rows                  = 1;
  h.chunks                = 2;
  const uint64_t keys[]   = {9, 9, 9, 4};
  const float gradients[] = {1.0e20f, -1.0e20f, 3.0f, 5.0f};
  const int arrivals[][4] = {{3, 1, 0, 2}, {2, 0, 3, 1}};
  int calls               = 0;
  auto apply              = [&](const auto& actual_keys, const auto& grads) {
    ++calls;
    EXPECT_EQ(actual_keys, std::vector<uint64_t>({9, 4}));
    EXPECT_EQ(grads, std::vector<float>({3, 5}));
  };
  for (h.step = 0; h.step < 2; ++h.step) {
    std::vector<std::string> packets;
    for (int index = 0; index < 4; ++index) {
      h.rank  = index / 2;
      h.chunk = index % 2;
      packets.push_back(
          PackSparseGradientCommit(h, keys + index, gradients + index));
    }
    for (int received = 0; received < 4; ++received) {
      const auto& packet  = packets[arrivals[h.step][received]];
      const auto expected = received == 3 ? h.step : h.step - 1;
      EXPECT_EQ(coordinator.Exchange("table", packet, apply), expected);
      EXPECT_EQ(coordinator.Exchange("table", packet, apply), expected);
      EXPECT_EQ(calls, h.step + (received == 3));
    }
  }
}

TEST(SparseGradientCommit, ChangedChunkCountsAndEmptyStepsResetAggregation) {
  SparseGradientCommitCoordinator coordinator;
  SparseGradientCommitHeader h;
  h.session  = 13;
  h.world    = 2;
  h.dim      = 1;
  h.divisor  = 2;
  int calls  = 0;
  auto apply = [&](const auto& keys, const auto& grads) {
    ++calls;
    EXPECT_EQ(keys, std::vector<uint64_t>({h.step == 0 ? 5u : 9u}));
    EXPECT_EQ(grads, std::vector<float>({h.step == 0 ? 7.0f : 2.0f}));
  };
  for (h.step = 0; h.step < 3; ++h.step) {
    for (h.rank = 0; h.rank < h.world; ++h.rank) {
      h.chunks = (h.step + h.rank) % 2 + 1;
      for (h.chunk = 0; h.chunk < h.chunks; ++h.chunk) {
        h.rows             = h.step != 1 && h.chunk == 0 ? 1 : 0;
        const uint64_t key = h.step == 0 ? 5 : 9;
        const float grad   = h.step == 0 ? 7 : 2;
        auto packet        = PackSparseGradientCommit(h, &key, &grad);
        const bool last    = h.rank + 1 == h.world && h.chunk + 1 == h.chunks;
        EXPECT_EQ(coordinator.Exchange("table", std::move(packet), apply),
                  last ? h.step : h.step - 1);
      }
    }
    EXPECT_EQ(calls, h.step == 0 ? 1 : h.step);
  }
}

TEST(SparseGradientCommit, AbortRejectsPendingCommitAndQueries) {
  SparseGradientCommitCoordinator coordinator;
  SparseGradientCommitHeader h;
  h.session          = 14;
  h.world            = 2;
  h.dim              = 1;
  h.rows             = 1;
  const uint64_t key = 3;
  const float grad   = 1;
  int calls          = 0;
  auto apply         = [&](const auto&, const auto&) { ++calls; };
  const auto packet  = PackSparseGradientCommit(h, &key, &grad);
  EXPECT_EQ(coordinator.Exchange("table", packet, apply), -1);
  h.rows = 0;
  h.kind = 3;
  EXPECT_THROW(
      coordinator.Exchange("table", PackSparseGradientCommit(h), apply),
      std::runtime_error);
  h.kind = 0;
  h.rank = 1;
  EXPECT_THROW(
      coordinator.Exchange("table", PackSparseGradientCommit(h), apply),
      std::runtime_error);
  EXPECT_THROW(coordinator.Exchange("table", packet, apply),
               std::runtime_error);
  h.kind = 1;
  EXPECT_THROW(
      coordinator.Exchange("table", PackSparseGradientCommit(h), apply),
      std::runtime_error);
  h.kind = 2;
  EXPECT_THROW(
      coordinator.Exchange("table", PackSparseGradientCommit(h), apply),
      std::runtime_error);
  EXPECT_EQ(calls, 0);
}
