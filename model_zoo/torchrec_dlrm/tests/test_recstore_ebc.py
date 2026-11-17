"""Numerical EBC regressions through native HierKV ops (no transport mocks).

Run from the repository root with PYTHONPATH=src/python/pytorch. These tests
cover the Python/autograd/native boundary; KVEngine and transport have separate
integration tests. Every scenario uses an independent torch.nn reference.
"""

import os
import unittest
import uuid

import torch
from torchrec.sparse.jagged_tensor import KeyedJaggedTensor

# CTest supplies the target from the active build directory.
if os.environ.get("RECSTORE_TEST_OPS_LIBRARY"):
    torch.ops.load_library(os.environ["RECSTORE_TEST_OPS_LIBRARY"])

from recstore.KVClient import RecStoreClient
from recstore.optimizer import SparseSGD
from torchrec_kv.EmbeddingBag import RecStoreEmbeddingBagCollection


class TestRecStoreEmbeddingBagCollection(unittest.TestCase):
    def setUp(self):
        self.saved_client = RecStoreClient._instance
        RecStoreClient._instance = None
        self.addCleanup(setattr, RecStoreClient, "_instance", self.saved_client)
        self.client = RecStoreClient()
        previous_backend = self.client.current_ps_backend()
        restore_backend = {"distributed_grpc": "grpc", "distributed_brpc": "brpc"}.get(
            previous_backend, previous_backend
        )
        self.addCleanup(self.client.set_ps_backend, restore_backend)
        self.client.set_ps_backend("hierkv")
        self.assertEqual(self.client.current_ps_backend(), "hierkv")
        self.generator = torch.Generator().manual_seed(20261118)

    def _check_training(self, *, fused, dims, prefetch=False):
        sizes = [37, 53]
        names = [f"ebc_{uuid.uuid4().hex}_{i}" for i in range(2)]
        # Two features share a table; the second table reuses the same local IDs.
        feature_tables = {"history": 0, "candidate": 1, "context": 0}
        configs = [
            dict(name=names[i], num_embeddings=sizes[i], embedding_dim=dims[i],
                 feature_names=[f for f, table in feature_tables.items() if table == i])
            for i in range(2)
        ]
        module = RecStoreEmbeddingBagCollection(
            configs, kv_client=self.client, enable_fusion=fused, fusion_k=6,
            fast_path_mode="off",
        )
        reference = torch.nn.ModuleList([
            torch.nn.EmbeddingBag(size, dim, mode="sum", include_last_offset=True)
            for size, dim in zip(sizes, dims)
        ])
        for i, table in enumerate(reference):
            with torch.no_grad():
                table.weight.copy_(torch.randn(table.weight.shape, generator=self.generator) * 0.2 + i)
            ids = torch.arange(sizes[i], dtype=torch.int64)
            self.client.push(names[0] if fused else names[i],
                             ids + (i * 64 if fused else 0), table.weight.detach())
        optimizer = SparseSGD([module], lr=0.07)
        oracle = torch.optim.SGD(reference.parameters(), lr=0.07)

        def check_weights():
            # Include rows never looked up, not just rows present in this batch.
            for i, table in enumerate(reference):
                ids = torch.arange(sizes[i], dtype=torch.int64)
                actual = self.client.pull(names[0] if fused else names[i],
                                          ids + (i * 64 if fused else 0))
                torch.testing.assert_close(actual, table.weight, rtol=1e-5, atol=1e-6)

        for step, batch_size in enumerate([5, 3, 7, 2]):
            with self.subTest(step=step, fused=fused, dims=dims, prefetch=prefetch):
                # Vary the learning rate to catch a backend silently using its default.
                lr = [0.07, 0.025, 0.11, 0.04][step]
                optimizer.param_groups[0]["lr"] = lr
                oracle.param_groups[0]["lr"] = lr
                optimizer.zero_grad()
                oracle.zero_grad()
                # Reorder input features independently of table configuration order.
                order = (["candidate", "context", "history"] if step % 2 == 0
                         else ["history", "candidate", "context"])
                values, lengths, expected, inputs = [], [], [], []
                for feature in order:
                    table_index = feature_tables[feature]
                    lens = torch.randint(1, 5, (batch_size,), generator=self.generator)
                    # Missing history in an otherwise populated batch is a real input.
                    if feature == "history":
                        lens[1] = 0
                    ids = torch.randint(0, 13, (int(lens.sum()),), generator=self.generator)
                    ids[::2] = 3  # Repeated hot ID within and across bags/features.
                    offsets = torch.cat([torch.zeros(1, dtype=torch.int64), lens.cumsum(0)])
                    expected.append(reference[table_index](ids, offsets))
                    inputs.append((table_index, ids, offsets))
                    values.append(ids)
                    lengths.append(lens)
                features = KeyedJaggedTensor.from_lengths_sync(
                    keys=order, values=torch.cat(values), lengths=torch.cat(lengths).to(torch.int32),
                )
                if prefetch:
                    module.issue_fused_prefetch(features)
                result = module(features)
                self.assertEqual(result.keys(), order)
                expected_output = torch.cat(expected, dim=1)
                torch.testing.assert_close(result.values(), expected_output, rtol=1e-5, atol=1e-6)
                # A nonuniform, weight-dependent gradient exposes wrong bag layout,
                # duplicate reduction, table routing, and stale reads in later steps.
                target = torch.randn(expected_output.shape, generator=self.generator)
                loss = (result.values() - target).square().mean()
                expected_loss = (expected_output - target).square().mean()
                torch.testing.assert_close(loss, expected_loss, rtol=1e-5, atol=1e-6)
                loss.backward()
                expected_loss.backward()
                check_weights()  # Backward records gradients; it must not update storage.
                optimizer.step()  # Submission.
                optimizer.flush()  # Completion before the next read or prefetch.
                oracle.step()
                check_weights()
                # The consumed prefetch must not hide this step's update.
                with torch.no_grad():
                    next_expected = torch.cat([reference[i](ids, offsets)
                                               for i, ids, offsets in inputs], dim=1)
                    torch.testing.assert_close(module(features).values(), next_expected,
                                               rtol=1e-5, atol=1e-6)
                # A second step with no new backward must not replay consumed gradients.
                optimizer.step()
                optimizer.flush()
                check_weights()

    def test_multitable_training_matches_pytorch(self):
        self._check_training(fused=False, dims=[16, 32])

    def test_fused_training_matches_pytorch(self):
        self._check_training(fused=True, dims=[16, 16])

    def test_fused_prefetch_training_matches_pytorch(self):
        self._check_training(fused=True, dims=[16, 16], prefetch=True)


if __name__ == "__main__":
    unittest.main(verbosity=2)
