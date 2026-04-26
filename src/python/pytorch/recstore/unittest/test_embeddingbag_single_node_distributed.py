import importlib
import sys
import types
import unittest

import torch


class _FakeEmbeddingBagConfig:
    def __init__(self, **kwargs):
        self.name = kwargs["name"]
        self.embedding_dim = kwargs["embedding_dim"]
        self.num_embeddings = kwargs["num_embeddings"]
        self.feature_names = kwargs["feature_names"]


class _FakeKeyedTensor:
    def __init__(self, *, keys, values, length_per_key):
        self._keys = list(keys)
        self._values = values
        self._length_per_key = list(length_per_key)

    def keys(self):
        return list(self._keys)

    def values(self):
        return self._values

    def length_per_key(self):
        return list(self._length_per_key)


class _FakeFeature:
    def __init__(self, values: torch.Tensor, lengths: torch.Tensor):
        self._values = values
        self._lengths = lengths

    def values(self):
        return self._values

    def lengths(self):
        return self._lengths


class _FakeKeyedJaggedTensor:
    def __init__(self, per_key):
        self._per_key = dict(per_key)

    def keys(self):
        return list(self._per_key.keys())

    def __getitem__(self, key):
        return self._per_key[key]

    def device(self):
        for feature in self._per_key.values():
            return feature.values().device
        return torch.device("cpu")


class _FakeKVClient:
    def __init__(self):
        self.init_data_calls = []
        self.pull_calls = []
        self.local_lookup_calls = []
        self.set_ps_backend_calls = []
        self.activate_shard_calls = []
        self.embedding_dim = 2

    def init_data(self, **kwargs):
        self.init_data_calls.append(kwargs)

    def pull(self, name, ids):
        ids = ids.to(torch.int64).cpu()
        self.pull_calls.append((name, ids.clone()))
        rows = [[float(v.item()), float(v.item()) + 0.5] for v in ids]
        return torch.tensor(rows, dtype=torch.float32)

    def local_lookup_flat(self, name, ids):
        ids = ids.to(torch.int64).cpu()
        self.local_lookup_calls.append((name, ids.clone()))
        rows = [[float(v.item()) + 100.0, float(v.item()) + 100.5] for v in ids]
        return torch.tensor(rows, dtype=torch.float32)

    def set_ps_backend(self, backend):
        self.set_ps_backend_calls.append(str(backend))

    def activate_shard(self, shard):
        self.activate_shard_calls.append(int(shard))

    def is_shared_local_shm_table(self):
        return False


class _FakeDist:
    def __init__(self, *, initialized, rank, world_size):
        self._initialized = initialized
        self._rank = rank
        self._world_size = world_size

    def is_available(self):
        return True

    def is_initialized(self):
        return self._initialized

    def get_rank(self):
        return self._rank

    def get_world_size(self):
        return self._world_size


def _install_torchrec_stub():
    saved = {
        name: sys.modules.get(name)
        for name in [
            "torchrec",
            "torchrec.sparse",
            "torchrec.sparse.jagged_tensor",
            "torchrec.modules",
            "torchrec.modules.embedding_configs",
        ]
    }

    torchrec_mod = types.ModuleType("torchrec")
    sparse_mod = types.ModuleType("torchrec.sparse")
    jagged_mod = types.ModuleType("torchrec.sparse.jagged_tensor")
    modules_mod = types.ModuleType("torchrec.modules")
    embedding_configs_mod = types.ModuleType("torchrec.modules.embedding_configs")

    jagged_mod.KeyedJaggedTensor = _FakeKeyedJaggedTensor
    jagged_mod.KeyedTensor = _FakeKeyedTensor
    embedding_configs_mod.EmbeddingBagConfig = _FakeEmbeddingBagConfig

    sys.modules["torchrec"] = torchrec_mod
    sys.modules["torchrec.sparse"] = sparse_mod
    sys.modules["torchrec.sparse.jagged_tensor"] = jagged_mod
    sys.modules["torchrec.modules"] = modules_mod
    sys.modules["torchrec.modules.embedding_configs"] = embedding_configs_mod
    return saved


def _restore_modules(saved):
    for name, module in saved.items():
        if module is None:
            sys.modules.pop(name, None)
        else:
            sys.modules[name] = module


class TestEmbeddingBagSingleNodeDistributed(unittest.TestCase):
    def setUp(self):
        self._saved_modules = _install_torchrec_stub()
        sys.modules.pop("src.python.pytorch.torchrec_kv.EmbeddingBag", None)
        self.embeddingbag_module = importlib.import_module(
            "src.python.pytorch.torchrec_kv.EmbeddingBag"
        )
        self._original_get_kv_client = self.embeddingbag_module.get_kv_client
        self.embeddingbag_module.get_kv_client = lambda: _FakeKVClient()
        self._original_dist = self.embeddingbag_module.torch.distributed

    def tearDown(self):
        self.embeddingbag_module.get_kv_client = self._original_get_kv_client
        self.embeddingbag_module.torch.distributed = self._original_dist
        _restore_modules(self._saved_modules)

    def _build_ebc(self):
        return self.embeddingbag_module.RecStoreEmbeddingBagCollection(
            [
                {
                    "name": "table0",
                    "embedding_dim": 2,
                    "num_embeddings": 128,
                    "feature_names": ["f1"],
                }
            ],
            enable_fusion=True,
        )

    def _build_features(self, values):
        return _FakeKeyedJaggedTensor(
            {
                "f1": _FakeFeature(
                    values=torch.tensor(values, dtype=torch.int64),
                    lengths=torch.tensor([len(values)], dtype=torch.int64),
                )
            }
        )

    def test_forward_keeps_legacy_pull_path_when_fast_path_disabled(self):
        ebc = self._build_ebc()
        ebc.enable_single_node_distributed_fast_path = False
        self.embeddingbag_module.torch.distributed = _FakeDist(
            initialized=True,
            rank=0,
            world_size=2,
        )

        features = self._build_features([3, 7])
        out = ebc(features)

        self.assertEqual(len(ebc.kv_client.pull_calls), 1)
        self.assertEqual(len(ebc.kv_client.local_lookup_calls), 0)
        self.assertTrue(
            torch.equal(
                out.values(),
                torch.tensor([[10.0, 11.0]], dtype=torch.float32),
            )
        )

    def test_forward_uses_owner_lookup_fast_path_when_explicitly_enabled(self):
        ebc = self._build_ebc()
        ebc.enable_single_node_distributed_fast_path = True
        ebc.single_node_distributed_mode = "single_node"
        self.embeddingbag_module.torch.distributed = _FakeDist(
            initialized=True,
            rank=0,
            world_size=2,
        )

        requests = []
        responses = []

        def fake_exchange_lookup_ids(payload, *, world_size, backend):
            requests.append((payload, world_size, backend))
            return [
                self.embeddingbag_module.LookupIdsPayload(
                    rank=0,
                    destination_ranks=torch.tensor([0, 0], dtype=torch.int64),
                    source_ranks=torch.tensor([0, 0], dtype=torch.int64),
                    row_positions=torch.tensor([0, 2], dtype=torch.int64),
                    fused_ids=torch.tensor([4, 8], dtype=torch.int64),
                ),
                self.embeddingbag_module.LookupIdsPayload(
                    rank=1,
                    destination_ranks=torch.tensor([], dtype=torch.int64),
                    source_ranks=torch.tensor([], dtype=torch.int64),
                    row_positions=torch.tensor([], dtype=torch.int64),
                    fused_ids=torch.tensor([], dtype=torch.int64),
                ),
            ]

        def fake_exchange_lookup_embedding_responses(payload, *, world_size, backend):
            responses.append((payload, world_size, backend))
            return [
                payload,
                self.embeddingbag_module.LookupEmbeddingResponsePayload(
                    rank=1,
                    requestor_ranks=torch.tensor([0], dtype=torch.int64),
                    row_positions=torch.tensor([1], dtype=torch.int64),
                    embeddings=torch.tensor([[107.0, 107.5]], dtype=torch.float32),
                ),
            ]

        original_exchange_lookup_ids = getattr(
            self.embeddingbag_module, "exchange_lookup_ids", None
        )
        original_exchange_lookup_embedding_responses = getattr(
            self.embeddingbag_module, "exchange_lookup_embedding_responses", None
        )
        self.embeddingbag_module.exchange_lookup_ids = fake_exchange_lookup_ids
        self.embeddingbag_module.exchange_lookup_embedding_responses = (
            fake_exchange_lookup_embedding_responses
        )
        try:
            features = self._build_features([4, 7, 8])
            out = ebc(features)
        finally:
            if original_exchange_lookup_ids is None:
                delattr(self.embeddingbag_module, "exchange_lookup_ids")
            else:
                self.embeddingbag_module.exchange_lookup_ids = original_exchange_lookup_ids
            if original_exchange_lookup_embedding_responses is None:
                delattr(self.embeddingbag_module, "exchange_lookup_embedding_responses")
            else:
                self.embeddingbag_module.exchange_lookup_embedding_responses = (
                    original_exchange_lookup_embedding_responses
                )

        self.assertEqual(len(requests), 1)
        self.assertEqual(len(responses), 1)
        self.assertEqual(len(ebc.kv_client.pull_calls), 0)
        self.assertEqual(len(ebc.kv_client.local_lookup_calls), 1)
        self.assertEqual(ebc.kv_client.set_ps_backend_calls, ["local_shm"])
        self.assertEqual(ebc.kv_client.activate_shard_calls, [0])
        self.assertTrue(
            torch.equal(
                ebc.kv_client.local_lookup_calls[0][1],
                torch.tensor([4, 8], dtype=torch.int64),
            )
        )
        self.assertTrue(
            torch.equal(
                out.values(),
                torch.tensor([[319.0, 320.5]], dtype=torch.float32),
            )
        )

    def test_forward_uses_direct_local_lookup_for_shared_local_shm_single_table(self):
        ebc = self._build_ebc()
        ebc.enable_single_node_distributed_fast_path = True
        ebc.single_node_distributed_mode = "single_node"
        ebc.kv_client.is_shared_local_shm_table = lambda: True
        self.embeddingbag_module.torch.distributed = _FakeDist(
            initialized=True,
            rank=0,
            world_size=2,
        )

        def fail_exchange_lookup_ids(*args, **kwargs):
            raise AssertionError("shared local_shm single-table path should bypass lookup id exchange")

        def fail_exchange_lookup_embedding_responses(*args, **kwargs):
            raise AssertionError("shared local_shm single-table path should bypass embedding response exchange")

        original_exchange_lookup_ids = getattr(
            self.embeddingbag_module, "exchange_lookup_ids", None
        )
        original_exchange_lookup_embedding_responses = getattr(
            self.embeddingbag_module, "exchange_lookup_embedding_responses", None
        )
        self.embeddingbag_module.exchange_lookup_ids = fail_exchange_lookup_ids
        self.embeddingbag_module.exchange_lookup_embedding_responses = (
            fail_exchange_lookup_embedding_responses
        )
        try:
            features = self._build_features([4, 7, 8])
            out = ebc(features)
        finally:
            if original_exchange_lookup_ids is None:
                delattr(self.embeddingbag_module, "exchange_lookup_ids")
            else:
                self.embeddingbag_module.exchange_lookup_ids = original_exchange_lookup_ids
            if original_exchange_lookup_embedding_responses is None:
                delattr(self.embeddingbag_module, "exchange_lookup_embedding_responses")
            else:
                self.embeddingbag_module.exchange_lookup_embedding_responses = (
                    original_exchange_lookup_embedding_responses
                )

        self.assertEqual(len(ebc.kv_client.pull_calls), 0)
        self.assertEqual(len(ebc.kv_client.local_lookup_calls), 1)
        self.assertEqual(ebc.kv_client.set_ps_backend_calls, ["local_shm"])
        self.assertEqual(ebc.kv_client.activate_shard_calls, [0])
        self.assertTrue(
            torch.equal(
                ebc.kv_client.local_lookup_calls[0][1],
                torch.tensor([4, 7, 8], dtype=torch.int64),
            )
        )
        self.assertTrue(
            torch.equal(
                out.values(),
                torch.tensor([[319.0, 320.5]], dtype=torch.float32),
            )
        )

    def test_forward_fast_path_restores_original_order_for_repeated_fused_ids(self):
        ebc = self._build_ebc()
        ebc.enable_single_node_distributed_fast_path = True
        ebc.single_node_distributed_mode = "single_node"
        self.embeddingbag_module.torch.distributed = _FakeDist(
            initialized=True,
            rank=0,
            world_size=2,
        )

        def fake_exchange_lookup_ids(payload, *, world_size, backend):
            return [
                self.embeddingbag_module.LookupIdsPayload(
                    rank=0,
                    destination_ranks=torch.tensor([0, 0], dtype=torch.int64),
                    source_ranks=torch.tensor([0, 0], dtype=torch.int64),
                    row_positions=torch.tensor([0, 2], dtype=torch.int64),
                    fused_ids=torch.tensor([6, 6], dtype=torch.int64),
                ),
                self.embeddingbag_module.LookupIdsPayload(
                    rank=1,
                    destination_ranks=torch.tensor([], dtype=torch.int64),
                    source_ranks=torch.tensor([], dtype=torch.int64),
                    row_positions=torch.tensor([], dtype=torch.int64),
                    fused_ids=torch.tensor([], dtype=torch.int64),
                ),
            ]

        def fake_exchange_lookup_embedding_responses(payload, *, world_size, backend):
            return [
                payload,
                self.embeddingbag_module.LookupEmbeddingResponsePayload(
                    rank=1,
                    requestor_ranks=torch.tensor([0, 0], dtype=torch.int64),
                    row_positions=torch.tensor([1, 3], dtype=torch.int64),
                    embeddings=torch.tensor(
                        [[701.0, 701.5], [703.0, 703.5]],
                        dtype=torch.float32,
                    ),
                ),
            ]

        original_exchange_lookup_ids = getattr(
            self.embeddingbag_module, "exchange_lookup_ids", None
        )
        original_exchange_lookup_embedding_responses = getattr(
            self.embeddingbag_module, "exchange_lookup_embedding_responses", None
        )
        self.embeddingbag_module.exchange_lookup_ids = fake_exchange_lookup_ids
        self.embeddingbag_module.exchange_lookup_embedding_responses = (
            fake_exchange_lookup_embedding_responses
        )
        try:
            features = self._build_features([6, 7, 6, 7])
            out = ebc(features)
        finally:
            if original_exchange_lookup_ids is None:
                delattr(self.embeddingbag_module, "exchange_lookup_ids")
            else:
                self.embeddingbag_module.exchange_lookup_ids = original_exchange_lookup_ids
            if original_exchange_lookup_embedding_responses is None:
                delattr(self.embeddingbag_module, "exchange_lookup_embedding_responses")
            else:
                self.embeddingbag_module.exchange_lookup_embedding_responses = (
                    original_exchange_lookup_embedding_responses
                )

        self.assertTrue(
            torch.equal(
                out.values(),
                torch.tensor([[1616.0, 1618.0]], dtype=torch.float32),
            )
        )

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA is required for GPU-resident lookup fast path coverage")
    def test_forward_fast_path_keeps_owner_lookup_ids_on_cuda(self):
        ebc = self._build_ebc()
        ebc.enable_single_node_distributed_fast_path = True
        ebc.single_node_distributed_mode = "single_node"
        self.embeddingbag_module.torch.distributed = _FakeDist(
            initialized=True,
            rank=0,
            world_size=2,
        )

        device = torch.device("cuda", 0)

        def fake_exchange_lookup_ids(payload, *, world_size, backend):
            return [
                self.embeddingbag_module.LookupIdsPayload(
                    rank=0,
                    destination_ranks=torch.tensor([0, 0], dtype=torch.int64, device=device),
                    source_ranks=torch.tensor([0, 0], dtype=torch.int64, device=device),
                    row_positions=torch.tensor([0, 2], dtype=torch.int64, device=device),
                    fused_ids=torch.tensor([4, 8], dtype=torch.int64, device=device),
                ),
                self.embeddingbag_module.LookupIdsPayload(
                    rank=1,
                    destination_ranks=torch.empty((0,), dtype=torch.int64, device=device),
                    source_ranks=torch.empty((0,), dtype=torch.int64, device=device),
                    row_positions=torch.empty((0,), dtype=torch.int64, device=device),
                    fused_ids=torch.empty((0,), dtype=torch.int64, device=device),
                ),
            ]

        def fake_exchange_lookup_embedding_responses(payload, *, world_size, backend):
            return [
                payload,
                self.embeddingbag_module.LookupEmbeddingResponsePayload(
                    rank=1,
                    requestor_ranks=torch.tensor([0], dtype=torch.int64, device=device),
                    row_positions=torch.tensor([1], dtype=torch.int64, device=device),
                    embeddings=torch.tensor([[107.0, 107.5]], dtype=torch.float32, device=device),
                ),
            ]

        original_exchange_lookup_ids = getattr(self.embeddingbag_module, "exchange_lookup_ids", None)
        original_exchange_lookup_embedding_responses = getattr(
            self.embeddingbag_module,
            "exchange_lookup_embedding_responses",
            None,
        )
        self.embeddingbag_module.exchange_lookup_ids = fake_exchange_lookup_ids
        self.embeddingbag_module.exchange_lookup_embedding_responses = (
            fake_exchange_lookup_embedding_responses
        )
        lookup_devices = []

        def record_local_lookup(name, ids):
            lookup_devices.append(ids.device.type)
            return torch.tensor(
                [[104.0, 104.5], [108.0, 108.5]],
                dtype=torch.float32,
                device=ids.device,
            )

        ebc.kv_client.local_lookup_flat = record_local_lookup
        try:
            features = _FakeKeyedJaggedTensor(
                {
                    "f1": _FakeFeature(
                        values=torch.tensor([4, 7, 8], dtype=torch.int64, device=device),
                        lengths=torch.tensor([3], dtype=torch.int64, device=device),
                    )
                }
            )
            out = ebc(features)
        finally:
            if original_exchange_lookup_ids is None:
                delattr(self.embeddingbag_module, "exchange_lookup_ids")
            else:
                self.embeddingbag_module.exchange_lookup_ids = original_exchange_lookup_ids
            if original_exchange_lookup_embedding_responses is None:
                delattr(self.embeddingbag_module, "exchange_lookup_embedding_responses")
            else:
                self.embeddingbag_module.exchange_lookup_embedding_responses = (
                    original_exchange_lookup_embedding_responses
                )

        self.assertEqual(lookup_devices, ["cuda"])
        self.assertEqual(out.values().device.type, "cuda")


if __name__ == "__main__":
    unittest.main()
