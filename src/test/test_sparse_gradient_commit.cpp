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
