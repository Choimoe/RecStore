import unittest
import torch

from ...torchrec.EmbeddingBag import RecStoreEmbeddingBagCollection
from torchrec.sparse.jagged_tensor import KeyedJaggedTensor


class _FakeOps:
    def __init__(self):
        self._store = {}  # id(int)->tensor(1,D)
        self._next_handle = 1
        self._prefetch_buf = {}

    def emb_write(self, keys: torch.Tensor, values: torch.Tensor):
        keys = keys.to(torch.int64).cpu().contiguous()
        values = values.to(torch.float32).cpu().contiguous()
        assert values.dim() == 2
        for i in range(keys.numel()):
            self._store[int(keys[i].item())] = values[i:i+1].clone()

    def emb_read(self, keys: torch.Tensor, embedding_dim: int) -> torch.Tensor:
        keys = keys.to(torch.int64).cpu().contiguous()
        out = torch.zeros((keys.numel(), int(embedding_dim)), dtype=torch.float32)
        for i in range(keys.numel()):
            kid = int(keys[i].item())
            if kid in self._store:
                row = self._store[kid]
                if row.size(1) != embedding_dim:
                    # simple reshape/pad-truncate to requested dim
                    if row.size(1) > embedding_dim:
                        out[i] = row[0, :embedding_dim]
                    else:
                        out[i, :row.size(1)] = row[0]
                else:
                    out[i] = row[0]
            else:
                # keep zeros for missing
                pass
        return out

    def emb_prefetch(self, keys: torch.Tensor) -> int:
        # For testing, precompute the result with default dim 4; the real wait will pass dim anyway.
        keys = keys.to(torch.int64).cpu().contiguous()
        handle = self._next_handle
        self._next_handle += 1
        self._prefetch_buf[handle] = keys.clone()
        return handle

    def emb_wait_result(self, handle: int, embedding_dim: int) -> torch.Tensor:
        keys = self._prefetch_buf.pop(int(handle))
        return self.emb_read(keys, embedding_dim)


class TestFusedPrefetch(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(0)

    def _build_features(self):
        # Batch size = 2
        # f1: lengths [2,1] -> values 3 ids
        # f2: lengths [1,2] -> values 3 ids
        keys = ["f1", "f2"]
        values_f1 = torch.tensor([1, 2, 3], dtype=torch.int64)
        lengths_f1 = torch.tensor([2, 1], dtype=torch.int32)
        values_f2 = torch.tensor([0, 4, 2], dtype=torch.int64)
        lengths_f2 = torch.tensor([1, 2], dtype=torch.int32)
        values = torch.cat([values_f1, values_f2], dim=0)
        lengths = torch.cat([lengths_f1, lengths_f2], dim=0)
        kjt = KeyedJaggedTensor.from_lengths_sync(keys=keys, values=values, lengths=lengths)
        return kjt

    def test_fused_prefetch_matches_sync(self):
        configs = [
            dict(name="t0", embedding_dim=4, num_embeddings=16, feature_names=["f1"]),
            dict(name="t1", embedding_dim=4, num_embeddings=16, feature_names=["f2"]),
        ]
        ebc = RecStoreEmbeddingBagCollection(configs, enable_fusion=True, fusion_k=30)
        # Patch ops with fake backend and reinitialize data to zeros
        fake = _FakeOps()
        ebc.kv_client.ops = fake
        # mirror init_data writes
        for idx, cfg in enumerate(configs):
            base_offset = (idx << 30)
            n, d = cfg["num_embeddings"], cfg["embedding_dim"]
            keys = torch.arange(n, dtype=torch.int64) + base_offset
            vals = torch.zeros((n, d), dtype=torch.float32)
            fake.emb_write(keys, vals)

        # Build features after initialization; rely on zero initialization for both tables.
        features = self._build_features()

        # Sync path (no prefetch)
        out_sync = ebc(features).values().detach().clone()

        # Fused prefetch path
        ebc.issue_fused_prefetch(features)
        out_prefetch = ebc(features).values().detach().clone()

        self.assertTrue(torch.allclose(out_sync, out_prefetch), "Fused prefetch output must match sync output (zero-init case)")

        stats = ebc.report_prefetch_stats(reset=True)
        self.assertGreaterEqual(stats.get("batches_prefetched", 0), 1)


if __name__ == "__main__":
    unittest.main()
