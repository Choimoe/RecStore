#include "optimization/bagpipe/cache_metadata.h"

#include <gtest/gtest.h>
#include <algorithm>
#include <set>

using recstore::optimization::bagpipe::CacheMetadata;

TEST(BagPipeMetadata, OversizedAdmissionRetainsFinalRowsAndLruOrder) {
  CacheMetadata metadata(2);
  const int64_t ids[]  = {3, 7, 9};
  const auto admission = metadata.Admit(ids, 3);
  EXPECT_EQ(admission.evicted, 1);
  EXPECT_EQ(metadata.Size(), 2);
  EXPECT_EQ(
      std::set<int64_t>(admission.slots.begin(), admission.slots.end()).size(),
      2);
  EXPECT_EQ(std::set<int64_t>(admission.rows.begin(), admission.rows.end()),
            std::set<int64_t>({1, 2}));
  for (size_t i = 0; i < admission.slots.size(); ++i) {
    auto hit = metadata.Lookup(&ids[admission.rows[i]], 1);
    ASSERT_EQ(hit.hit_slots.size(), 1);
    EXPECT_EQ(hit.hit_slots[0], admission.slots[i]);
  }
  const int64_t touch = 7, next = 11;
  metadata.Lookup(&touch, 1);
  metadata.Admit(&next, 1);
  const int64_t query[] = {3, 7, 9, 11};
  auto result           = metadata.Lookup(query, 4);
  EXPECT_EQ(result.hit_positions, std::vector<int64_t>({1, 3}));
  EXPECT_EQ(result.missing_positions, std::vector<int64_t>({0, 2}));
}

TEST(BagPipeMetadata, CommitInvalidatesRemoteAndSharedRowsAndReusesSlots) {
  CacheMetadata metadata(3);
  const int64_t active[] = {3, 7, 9}, local[] = {3, 7};
  const bool exclusive[] = {true, false};
  metadata.Admit(active, 3);
  auto collective = metadata.Commit(active, 3, local, 2, exclusive, false);
  EXPECT_EQ(collective.invalidated, 0);
  EXPECT_EQ(collective.rows, std::vector<int64_t>({0, 1, 2}));
  auto ps = metadata.Commit(active, 3, local, 2, exclusive, true);
  EXPECT_EQ(ps.invalidated, 2);
  EXPECT_EQ(ps.rows, std::vector<int64_t>({0}));
  EXPECT_EQ(metadata.Size(), 1);
  const int64_t next[] = {11, 13};
  auto admission       = metadata.Admit(next, 2);
  EXPECT_EQ(admission.evicted, 0);
  auto lookup = metadata.Lookup(next, 2);
  EXPECT_EQ(lookup.hit_positions, std::vector<int64_t>({0, 1}));
  EXPECT_NE(lookup.hit_slots[0], lookup.hit_slots[1]);
  EXPECT_EQ(metadata.Size(), 3);
}

TEST(BagPipeMetadata,
     PrefetchRequiresCommittedPredecessorsAndMissingResidency) {
  CacheMetadata metadata(2);
  const int64_t local[] = {3, 9}, active[] = {3, 7};
  EXPECT_EQ(metadata.Previous(0, local, 2, active, 2),
            std::vector<int64_t>({-1, -1}));
  auto previous = metadata.Previous(1, local, 2, local, 2);
  EXPECT_EQ(previous, std::vector<int64_t>({0, -1}));
  EXPECT_EQ(metadata.Prefetch(local, previous.data(), 2, -1),
            std::vector<int64_t>({9}));
  EXPECT_EQ(metadata.Prefetch(local, previous.data(), 2, 0),
            std::vector<int64_t>({3, 9}));
  metadata.Admit(&local[1], 1);
  EXPECT_EQ(metadata.Prefetch(local, previous.data(), 2, 0),
            std::vector<int64_t>({3}));
  auto updated = metadata.UpdatedIds();
  std::sort(updated.begin(), updated.end());
  EXPECT_EQ(updated, std::vector<int64_t>({3, 7, 9}));
  metadata.ClearPlan();
  EXPECT_TRUE(metadata.UpdatedIds().empty());
  EXPECT_EQ(metadata.Size(), 1);
}

TEST(BagPipeMetadata, EmptyBatchesAndInvalidCapacity) {
  EXPECT_THROW(CacheMetadata(0), std::invalid_argument);
  EXPECT_THROW(CacheMetadata(-1), std::invalid_argument);
  CacheMetadata metadata(1);
  EXPECT_TRUE(metadata.Previous(0, nullptr, 0, nullptr, 0).empty());
  EXPECT_TRUE(metadata.Prefetch(nullptr, nullptr, 0, -1).empty());
  EXPECT_TRUE(metadata.Lookup(nullptr, 0).missing_positions.empty());
  EXPECT_TRUE(metadata.Admit(nullptr, 0).slots.empty());
  EXPECT_EQ(metadata.Commit(nullptr, 0, nullptr, 0, nullptr, true).invalidated,
            0);
}
