from __future__ import annotations

import contextlib
import io
import csv
from contextlib import ExitStack
import tempfile
import types
import unittest
from pathlib import Path
from unittest import mock

import torch

from model_zoo.rs_demo import config
from model_zoo.rs_demo.config import RunConfig
from model_zoo.rs_demo.runners import recstore_runner
from model_zoo.rs_demo.runners.recstore_runner import (
    LookaheadPrefetcher,
    RecStoreRunner,
    _build_train_dataloader_for_mode,
    _maybe_wrap_dense_module_for_dist,
)


class _DummyDense(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.linear = torch.nn.Linear(17, 1)

    def forward(self, dense_features: torch.Tensor, embedded_sparse: torch.Tensor) -> torch.Tensor:
        flat_sparse = embedded_sparse.reshape(embedded_sparse.shape[0], -1)
        features = torch.cat([dense_features, flat_sparse], dim=1).to(self.linear.weight.device)
        return self.linear(features)


class _FakeShardedClient:
    def __init__(self) -> None:
        self.emb_read_calls = 0
        self.emb_read_prefetch_calls = 0
        self.emb_prefetch_calls = 0
        self.emb_wait_result_calls = 0
        self.init_embedding_table_calls = 0
        self.emb_write_calls = 0
        self.set_ps_backend_calls: list[str] = []
        self.activate_shard_calls: list[int] = []
        self.enable_gpu_cache_calls: list[tuple[int, int]] = []
        self.enable_gpu_cache_result = True
        self.gpu_cache_lookup_bypass_enabled: bool | None = True
        self.gpu_cache_profile = {}
        self.local_shm_warmup_calls = 0
        self._shared_local_shm_table = False
        self._last_prefetch_keys = torch.empty((0,), dtype=torch.int64)
        self._current_ps_backend = "local_shm"

    def set_ps_backend(self, backend: str) -> None:
        backend = str(backend)
        self.set_ps_backend_calls.append(backend)
        self._current_ps_backend = backend

    def current_ps_backend(self) -> str:
        return self._current_ps_backend

    def activate_shard(self, shard: int) -> None:
        self.activate_shard_calls.append(int(shard))

    def enable_gpu_cache(self, capacity: int, embedding_dim: int) -> bool:
        self.enable_gpu_cache_calls.append((int(capacity), int(embedding_dim)))
        return self.enable_gpu_cache_result

    def set_gpu_cache_lookup_bypass_enabled(self, enabled: bool) -> None:
        self.gpu_cache_lookup_bypass_enabled = bool(enabled)

    def get_last_gpu_cache_profile(self):
        return self.gpu_cache_profile

    def init_embedding_table(self, table_name: str, num_embeddings: int, embedding_dim: int) -> bool:
        self.init_embedding_table_calls += 1
        return True

    def emb_write(self, keys: torch.Tensor, values: torch.Tensor) -> None:
        self.emb_write_calls += 1
        return None

    def emb_read(self, keys: torch.Tensor, embedding_dim: int) -> torch.Tensor:
        self.emb_read_calls += 1
        raise AssertionError("prefetch read mode should not call emb_read")

    def emb_read_prefetch(self, keys: torch.Tensor, embedding_dim: int) -> torch.Tensor:
        self.emb_read_prefetch_calls += 1
        return torch.zeros((keys.numel(), embedding_dim), dtype=torch.float32)

    def emb_prefetch(self, keys: torch.Tensor) -> int:
        self.emb_prefetch_calls += 1
        raise AssertionError("prefetch read mode should use stable emb_read_prefetch")

    def emb_wait_result(self, prefetch_id: int, embedding_dim: int) -> torch.Tensor:
        self.emb_wait_result_calls += 1
        raise AssertionError("prefetch read mode should use stable emb_read_prefetch")

    def emb_update_table(self, table_name: str, keys: torch.Tensor, grads: torch.Tensor) -> None:
        return None

    def warmup_local_lookup_flat_cuda_region(self) -> bool:
        self.local_shm_warmup_calls += 1
        return True

    def is_shared_local_shm_table(self) -> bool:
        return self._shared_local_shm_table


class _FakeDirectReadShardedClient(_FakeShardedClient):
    def emb_read(self, keys: torch.Tensor, embedding_dim: int) -> torch.Tensor:
        self.emb_read_calls += 1
        return torch.zeros((keys.numel(), embedding_dim), dtype=torch.float32)


class _FakeRecStoreEmbeddingBagCollection:
    last_instance = None

    def __init__(self, *args, **kwargs) -> None:
        self.args = args
        self.kwargs = kwargs
        self.kv_client = kwargs.get("kv_client")
        self.issue_fused_prefetch_calls = 0
        self.issue_fused_prefetch_record_flags: list[bool] = []
        self.set_fused_prefetch_handle_calls = 0
        self.reset_perf_stats_calls = 0
        self._single_node_forward_profile = {}
        _FakeRecStoreEmbeddingBagCollection.last_instance = self

    def issue_fused_prefetch(self, features, *, record_handle: bool = True):
        del features
        self.issue_fused_prefetch_calls += 1
        self.issue_fused_prefetch_record_flags.append(bool(record_handle))
        return (
            1000 + self.issue_fused_prefetch_calls,
            7,
            12.5,
            torch.tensor([1], dtype=torch.int64),
            torch.tensor([0], dtype=torch.int64),
        )

    def set_fused_prefetch_handle(self, *args, **kwargs) -> None:
        del args, kwargs
        self.set_fused_prefetch_handle_calls += 1

    def reset_perf_stats(self) -> None:
        self.reset_perf_stats_calls += 1

    def consume_perf_stats(self, reset: bool = True):
        del reset
        return {
            "prefetch_issue_ms": 0.2,
            "lookup_wait_ms": 0.6,
            "lookup_owner_exchange_ms": 0.4,
            "lookup_local_lookup_ms": 0.5,
            "lookup_reassemble_ms": 0.3,
            "pool_embedding_bag_ms": 0.7,
        }

    def __call__(self, features):
        return object()

    def _can_use_single_node_distributed_fast_path(self) -> bool:
        return bool(
            getattr(self, "enable_single_node_distributed_fast_path", False)
            and getattr(self, "single_node_distributed_mode", None) == "single_node"
        )


class _FakePrefetchModule:
    def __init__(self) -> None:
        self.next_handle = 100
        self.issued: list[tuple[object, bool]] = []
        self.consumed: list[tuple[int, int]] = []

    def issue_fused_prefetch(self, features, *, record_handle: bool = True):
        self.issued.append((features, record_handle))
        handle = self.next_handle
        self.next_handle += 1
        return (
            handle,
            int(getattr(features, "num_ids", 0)),
            10.0 + handle,
            torch.tensor([handle], dtype=torch.int64),
            torch.tensor([0], dtype=torch.int64),
        )

    def set_fused_prefetch_handle(self, handle, num_ids=None, issue_ts=None, **kwargs) -> None:
        del issue_ts, kwargs
        self.consumed.append((int(handle), int(num_ids or 0)))


class _FakeSparseFeatures:
    def __init__(self, num_ids: int) -> None:
        self.num_ids = int(num_ids)


class _FakeSparseSGD:
    last_instance = None

    def __init__(self, params, lr: float) -> None:
        self.params = params
        self.lr = lr
        self.step_calls = 0
        self.flush_calls = 0
        self.zero_grad_calls = 0
        self.reset_perf_stats_calls = 0
        self._last_step_profile = {}
        _FakeSparseSGD.last_instance = self

    def zero_grad(self):
        self.zero_grad_calls += 1

    def step(self):
        self.step_calls += 1

    def flush(self):
        self.flush_calls += 1

    def reset_perf_stats(self) -> None:
        self.reset_perf_stats_calls += 1

    def consume_perf_stats(self, reset: bool = True):
        del reset
        return {
            "update_trace_merge_ms": 0.25,
            "update_owner_exchange_ms": 0.35,
            "update_local_apply_ms": 0.45,
            "update_async_enqueue_ms": 0.05,
            "update_flush_wait_ms": 0.15,
        }


class _FakeDenseOptimizer:
    def __init__(self, params, lr: float) -> None:
        self.params = list(params)
        self.lr = float(lr)
        self.zero_grad_calls = 0
        self.step_calls = 0

    def zero_grad(self, *args, **kwargs) -> None:
        del args, kwargs
        self.zero_grad_calls += 1

    def step(self) -> None:
        self.step_calls += 1


class TestRecStoreRunner(unittest.TestCase):
    def setUp(self) -> None:
        self._append_worker_debug_patch = mock.patch(
            "model_zoo.rs_demo.runners.recstore_runner._append_worker_debug",
            lambda *args, **kwargs: None,
        )
        self._append_worker_debug_patch.start()
        self.addCleanup(self._append_worker_debug_patch.stop)

    def test_lookahead_prefetcher_depth_zero_never_issues_prefetch(self) -> None:
        module = _FakePrefetchModule()
        prefetcher = LookaheadPrefetcher(module, depth=0)

        prefetcher.enqueue(_FakeSparseFeatures(3))
        prefetcher.attach_next()
        stats = prefetcher.consume_stats()

        self.assertEqual(module.issued, [])
        self.assertEqual(module.consumed, [])
        self.assertEqual(stats["prefetch_depth"], 0)
        self.assertEqual(stats["prefetch_issued_batches"], 0)
        self.assertEqual(stats["prefetch_consumed_batches"], 0)

    def test_lookahead_prefetcher_delays_consumption_by_depth(self) -> None:
        module = _FakePrefetchModule()
        prefetcher = LookaheadPrefetcher(module, depth=2)

        prefetcher.enqueue(_FakeSparseFeatures(3))
        self.assertFalse(prefetcher.attach_next())

        prefetcher.enqueue(_FakeSparseFeatures(5))
        self.assertFalse(prefetcher.attach_next())

        prefetcher.enqueue(_FakeSparseFeatures(7))
        self.assertTrue(prefetcher.advance())
        self.assertTrue(prefetcher.attach_next())
        stats = prefetcher.consume_stats()

        self.assertEqual([item[1] for item in module.issued], [False, False, False])
        self.assertEqual(module.consumed, [(100, 3)])
        self.assertEqual(stats["prefetch_depth"], 2)
        self.assertEqual(stats["prefetch_issued_batches"], 3)
        self.assertEqual(stats["prefetch_consumed_batches"], 1)
        self.assertEqual(stats["prefetch_pending_batches"], 2)
        self.assertEqual(stats["prefetch_total_ids"], 15)
        self.assertEqual(stats["prefetch_consumed_total_ids"], 3)
        self.assertGreater(stats["prefetch_issue_to_consume_ms"], 0)

    def test_warmup_gpu_local_shm_fast_path_runs_only_for_shared_cuda_fast_path(self) -> None:
        cfg = RunConfig(
            backend="recstore",
            enable_single_node_distributed_fast_path=True,
        )
        client = _FakeShardedClient()
        client._shared_local_shm_table = True

        warmed = recstore_runner._maybe_warmup_gpu_local_shm_fast_path(
            cfg=cfg,
            client=client,
            device=torch.device("cuda:0"),
        )

        self.assertTrue(warmed)
        self.assertEqual(client.local_shm_warmup_calls, 1)

    def test_warmup_gpu_local_shm_fast_path_skips_hierkv_backend(self) -> None:
        cfg = RunConfig(
            backend="recstore",
            enable_single_node_distributed_fast_path=True,
            single_node_ps_backend="hierkv",
        )
        client = _FakeShardedClient()
        client._shared_local_shm_table = True
        client.set_ps_backend("hierkv")

        warmed = recstore_runner._maybe_warmup_gpu_local_shm_fast_path(
            cfg=cfg,
            client=client,
            device=torch.device("cuda:0"),
        )

        self.assertFalse(warmed)
        self.assertEqual(client.current_ps_backend(), "hierkv")
        self.assertEqual(client.local_shm_warmup_calls, 0)

    def test_warmup_gpu_local_shm_fast_path_skips_when_conditions_do_not_match(self) -> None:
        cfg = RunConfig(
            backend="recstore",
            enable_single_node_distributed_fast_path=False,
        )
        client = _FakeShardedClient()
        client._shared_local_shm_table = True

        warmed = recstore_runner._maybe_warmup_gpu_local_shm_fast_path(
            cfg=cfg,
            client=client,
            device=torch.device("cuda:0"),
        )

        self.assertFalse(warmed)
        self.assertEqual(client.local_shm_warmup_calls, 0)

    def _run_local_worker_with_fake_embedding_module(
        self,
        cfg: RunConfig,
        *,
        rank: int = 0,
        world_size: int = 1,
        local_rank: int = 0,
    ) -> _FakeRecStoreEmbeddingBagCollection:
        runner_runtime = Path(tempfile.mkdtemp())
        repo_root = Path("/app/RecStore")

        dense = torch.zeros((1, 13), dtype=torch.float32)
        sparse = torch.zeros((1, 1), dtype=torch.int64)
        labels = torch.zeros((1, 1), dtype=torch.float32)
        dataset = [(dense, sparse, labels)]
        dataloader = [(dense, sparse, labels)]

        fake_client = _FakeDirectReadShardedClient()
        fake_client_module = types.ModuleType("client")
        fake_client_module.RecstoreClient = lambda library_path=None: object()
        fake_embeddingbag_module = types.ModuleType("python.pytorch.torchrec_kv.EmbeddingBag")
        class _ProfiledEmbeddingBagCollection(_FakeRecStoreEmbeddingBagCollection):
            def __init__(self, *args, **kwargs) -> None:
                super().__init__(*args, **kwargs)
                self._single_node_forward_profile = {
                    "lookup_local_lookup_ms": 1.25,
                    "lookup_wait_ms": 0.75,
                }

        fake_embeddingbag_module.RecStoreEmbeddingBagCollection = _ProfiledEmbeddingBagCollection
        fake_optimizer_module = types.ModuleType("python.pytorch.recstore.optimizer")
        class _ProfiledSparseSGD(_FakeSparseSGD):
            def __init__(self, params, lr: float) -> None:
                super().__init__(params, lr)
                self._last_step_profile = {
                    "exchange_ms": 2.5,
                    "local_update_ms": 3.5,
                    "local_update_backend_call_ms": 3.25,
                    "trace_collect_ms": 0.5,
                    "trace_aggregate_ms": 1.5,
                }

        fake_optimizer_module.SparseSGD = _ProfiledSparseSGD

        _FakeRecStoreEmbeddingBagCollection.last_instance = None

        with ExitStack() as stack:
            stack.enter_context(
                mock.patch.dict(
                    "sys.modules",
                    {
                        "client": fake_client_module,
                        "python.pytorch.torchrec_kv.EmbeddingBag": fake_embeddingbag_module,
                        "python.pytorch.recstore.optimizer": fake_optimizer_module,
                    },
                )
            )
            stack.enter_context(
                mock.patch("model_zoo.rs_demo.runners.recstore_runner.inject_project_paths", lambda *_: None)
            )
            stack.enter_context(
                mock.patch("model_zoo.rs_demo.runners.recstore_runner.torch.manual_seed", lambda *_: None)
            )
            stack.enter_context(
                mock.patch(
                    "model_zoo.rs_demo.runners.recstore_runner.torch.optim.SGD",
                    _FakeDenseOptimizer,
                )
            )
            stack.enter_context(
                mock.patch(
                    "model_zoo.rs_demo.runners.recstore_runner.detect_library_path",
                    lambda *_: repo_root / "build/lib/lib_recstore_ops.so",
                )
            )
            stack.enter_context(
                mock.patch(
                    "model_zoo.rs_demo.runners.recstore_runner.ShardedRecstoreClient",
                    lambda raw_client, runtime_dir: fake_client,
                )
            )
            stack.enter_context(
                mock.patch(
                    "model_zoo.rs_demo.runners.recstore_runner.get_default_cat_names",
                    lambda: ["cat_0"],
                )
            )
            stack.enter_context(
                mock.patch(
                    "model_zoo.rs_demo.runners.recstore_runner.build_train_dataloader",
                    lambda **kwargs: (dataset, dataloader),
                )
            )
            stack.enter_context(
                mock.patch(
                    "model_zoo.rs_demo.runners.recstore_runner.build_kjt_batch_from_dense_sparse_labels",
                    lambda *args, **kwargs: (None, object()),
                )
            )
            stack.enter_context(
                mock.patch(
                    "model_zoo.rs_demo.runners.recstore_runner.build_hybrid_dense_arch",
                    lambda *args, **kwargs: _DummyDense().to(kwargs["device"]),
                )
            )
            stack.enter_context(
                mock.patch(
                    "model_zoo.rs_demo.runners.recstore_runner.reshape_torchrec_embeddings_for_dlrm",
                    lambda **kwargs: torch.zeros((1, 1, 4), dtype=torch.float32, requires_grad=True),
                )
            )
            stack.enter_context(
                mock.patch(
                    "model_zoo.rs_demo.runners.recstore_runner.prepare_hybrid_dlrm_input",
                    lambda **kwargs: (
                        torch.zeros((1, 13), dtype=torch.float32, device=kwargs["device"]),
                        torch.zeros((1, 1, 4), dtype=torch.float32, device=kwargs["device"], requires_grad=True),
                        torch.zeros((1, 1), dtype=torch.float32, device=kwargs["device"]),
                    ),
                )
            )
            stack.enter_context(
                mock.patch(
                    "model_zoo.rs_demo.runners.recstore_runner.run_hybrid_backward",
                    lambda **kwargs: torch.zeros((1, 1, 4), dtype=torch.float32),
                )
            )
            stack.enter_context(
                mock.patch(
                    "model_zoo.rs_demo.runners.recstore_runner.sync_device",
                    lambda *args, **kwargs: None,
                )
            )
            stack.enter_context(
                mock.patch(
                    "model_zoo.rs_demo.runners.recstore_runner.finalize_recstore_row",
                    lambda row: row,
                )
            )
            stack.enter_context(
                mock.patch(
                    "model_zoo.rs_demo.runners.recstore_runner.summarize_us",
                    lambda xs: "ok",
                )
            )
            stack.enter_context(
                mock.patch(
                    "model_zoo.rs_demo.runners.recstore_runner.write_stage_csv",
                    lambda *args, **kwargs: None,
                )
            )

            runner = RecStoreRunner(runner_runtime)
            runner._run_local_worker(
                repo_root=repo_root,
                cfg=cfg,
                rank=rank,
                world_size=world_size,
                local_rank=local_rank,
                out_csv=runner_runtime / "rank.csv",
            )

        fake_ebc = _FakeRecStoreEmbeddingBagCollection.last_instance
        self.assertIsNotNone(fake_ebc)
        return fake_ebc

    def test_parse_config_keeps_single_node_fast_path_disabled_by_default(self) -> None:
        cfg = config.parse_config(["--backend", "recstore"])

        self.assertFalse(cfg.enable_single_node_distributed_fast_path)
        self.assertEqual(cfg.single_node_ps_backend, "local_shm")

    def test_parse_config_accepts_rdma_ps_type(self) -> None:
        cfg = config.parse_config(
            [
                "--backend",
                "recstore",
                "--ps-type",
                "RDMA",
                "--rdma-thread-num",
                "2",
                "--rdma-max-keys-per-request",
                "4096",
            ]
        )

        self.assertEqual(cfg.ps_type, "RDMA")
        self.assertEqual(cfg.rdma_thread_num, 2)
        self.assertEqual(cfg.rdma_max_keys_per_request, 4096)
        self.assertEqual(cfg.single_node_owner_policy, "hash_mod_world_size")

    def test_parse_config_accepts_gpu_cache_options(self) -> None:
        cfg = config.parse_config(
            [
                "--backend",
                "recstore",
                "--enable-gpu-cache",
                "--gpu-cache-capacity",
                "1024",
                "--disable-gpu-cache-lookup-bypass",
            ]
        )

        self.assertTrue(cfg.enable_gpu_cache)
        self.assertEqual(cfg.gpu_cache_capacity, 1024)
        self.assertTrue(cfg.disable_gpu_cache_lookup_bypass)

    def test_parse_config_accepts_prefetch_depth(self) -> None:
        cfg = config.parse_config(
            [
                "--backend",
                "recstore",
                "--prefetch-depth",
                "4",
            ]
        )

        self.assertEqual(cfg.prefetch_depth, 4)

    def test_validate_recstore_config_rejects_gpu_cache_without_capacity(self) -> None:
        cfg = RunConfig(backend="recstore")
        cfg.enable_gpu_cache = True
        cfg.gpu_cache_capacity = 0

        with self.assertRaisesRegex(
            RuntimeError,
            "--gpu-cache-capacity must be positive",
        ):
            config.validate_recstore_config(cfg)

    def test_validate_recstore_config_rejects_negative_prefetch_depth(self) -> None:
        cfg = RunConfig(backend="recstore", prefetch_depth=-1)

        with self.assertRaisesRegex(RuntimeError, "--prefetch-depth must be non-negative"):
            config.validate_recstore_config(cfg)

    def test_validate_recstore_config_allows_single_node_fast_path(self) -> None:
        cfg = RunConfig(
            backend="recstore",
            nnodes=1,
            nproc_per_node=2,
            enable_single_node_distributed_fast_path=True,
            single_node_ps_backend="local_shm",
            single_node_owner_policy="hash_mod_world_size",
        )

        config.validate_recstore_config(cfg)

    def test_validate_recstore_config_allows_hierkv_single_node_fast_path(self) -> None:
        cfg = RunConfig(
            backend="recstore",
            nnodes=1,
            nproc_per_node=2,
            enable_single_node_distributed_fast_path=True,
            single_node_ps_backend="hierkv",
            single_node_owner_policy="hash_mod_world_size",
        )

        config.validate_recstore_config(cfg)

    def test_validate_recstore_config_rejects_single_node_fast_path_with_multiple_nodes(self) -> None:
        cfg = RunConfig(
            backend="recstore",
            nnodes=2,
            nproc_per_node=2,
            node_rank=0,
            recstore_runtime_dir="/tmp/shared",
            enable_single_node_distributed_fast_path=True,
        )

        with self.assertRaisesRegex(
            RuntimeError,
            "single-node distributed fast path requires --nnodes=1",
        ):
            config.validate_recstore_config(cfg)

    def test_validate_recstore_config_rejects_single_node_fast_path_without_multiple_processes(self) -> None:
        cfg = RunConfig(
            backend="recstore",
            nnodes=1,
            nproc_per_node=1,
            enable_single_node_distributed_fast_path=True,
        )

        with self.assertRaisesRegex(
            RuntimeError,
            "single-node distributed fast path requires --nproc-per-node greater than 1",
        ):
            config.validate_recstore_config(cfg)

    def test_validate_recstore_config_rejects_invalid_fast_path_backend(self) -> None:
        cfg = RunConfig(
            backend="recstore",
            nnodes=1,
            nproc_per_node=2,
            enable_single_node_distributed_fast_path=True,
            single_node_ps_backend="brpc",
        )

        with self.assertRaisesRegex(
            RuntimeError,
            "single-node distributed fast path only supports --single-node-ps-backend=local_shm or hierkv",
        ):
            config.validate_recstore_config(cfg)

    def test_parse_config_rejects_invalid_single_node_owner_policy_choice(self) -> None:
        stderr = io.StringIO()
        with contextlib.redirect_stderr(stderr):
            with self.assertRaises(SystemExit):
                config.parse_config(
                    [
                        "--single-node-owner-policy",
                        "invalid_policy",
                    ]
                )

    def test_parse_config_rejects_invalid_single_node_ps_backend_choice(self) -> None:
        stderr = io.StringIO()
        with contextlib.redirect_stderr(stderr):
            with self.assertRaises(SystemExit):
                config.parse_config(
                    [
                        "--single-node-ps-backend",
                        "invalid_backend",
                    ]
                )

    def test_build_worker_fingerprint_includes_cli(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            repo_root = Path(tmpdir)
            for rel_path in (
                "model_zoo/rs_demo/cli.py",
                "model_zoo/rs_demo/config.py",
                "model_zoo/rs_demo/runners/recstore_runner.py",
                "model_zoo/rs_demo/runtime/hybrid_dlrm.py",
            ):
                path = repo_root / rel_path
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text(rel_path, encoding="utf-8")

            fingerprint = recstore_runner._build_worker_fingerprint(repo_root)

        self.assertIn("files", fingerprint)
        self.assertIn("model_zoo/rs_demo/cli.py", fingerprint["files"])

    def test_wrap_dense_module_for_dist_uses_ddp_when_distributed(self) -> None:
        module = _DummyDense()
        wrapped = object()

        with mock.patch(
            "torch.nn.parallel.DistributedDataParallel",
            return_value=wrapped,
        ) as ddp_ctor:
            result = _maybe_wrap_dense_module_for_dist(
                dense_module=module,
                device=torch.device("cpu"),
                local_rank=0,
                use_dist=True,
            )

        self.assertIs(result, wrapped)
        ddp_ctor.assert_called_once_with(module)

    def test_build_train_dataloader_for_distributed_uses_rank_partition(self) -> None:
        fake_dataset = [1, 2, 3]

        with mock.patch(
            "model_zoo.rs_demo.runners.recstore_runner.build_train_dataloader",
            return_value=(fake_dataset, "loader"),
        ) as build_loader:
            dataset, dataloader = _build_train_dataloader_for_mode(
                repo_root=Path("/app/RecStore"),
                cfg=RunConfig(
                    backend="recstore",
                    steps=1,
                    nnodes=2,
                    nproc_per_node=1,
                    batch_size=256,
                ),
                rank=1,
            )

        self.assertEqual(dataset, fake_dataset)
        self.assertEqual(dataloader, "loader")
        self.assertEqual(build_loader.call_args.kwargs["seed"], 20260330)
        self.assertEqual(build_loader.call_args.kwargs["shuffle"], True)
        self.assertEqual(build_loader.call_args.kwargs["rank"], 1)
        self.assertEqual(build_loader.call_args.kwargs["world_size"], 2)

    def test_runner_uses_world_size_from_nnodes_and_nproc_per_node(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            runtime_dir = Path(tmpdir)
            runner = RecStoreRunner(runtime_dir=runtime_dir)
            cfg = config.RunConfig(
                backend="recstore",
                nnodes=1,
                node_rank=0,
                nproc_per_node=2,
                output_root=tmpdir,
                run_id="recstore-dist",
            )

            with mock.patch.object(
                runner,
                "_run_distributed",
                return_value={"backend": "recstore", "rows": []},
            ) as dist_run:
                result = runner.run(Path(tmpdir), cfg)

            self.assertEqual(result["backend"], "recstore")
            dist_run.assert_called_once_with(Path(tmpdir), cfg)

    def test_runner_builds_torchrun_command_with_hybrid_arch_args(self) -> None:
        runner = RecStoreRunner(Path("/tmp/runtime"))
        cfg = RunConfig(
            backend="recstore",
            nnodes=1,
            node_rank=0,
            nproc_per_node=2,
            master_addr="127.0.0.1",
            master_port=29653,
            rdzv_backend="c10d",
            rdzv_id="recstore-case",
            output_root="/nas/home/shq/docker/rs_demo",
            run_id="recstore-case",
            recstore_main_csv="/nas/home/shq/docker/rs_demo/outputs/recstore-case/recstore_main.csv",
            recstore_main_agg_csv="/nas/home/shq/docker/rs_demo/outputs/recstore-case/recstore_main_agg.csv",
            dense_arch_layer_sizes="64,32,16",
            over_arch_layer_sizes="128,64,1",
        )

        cmd = runner._build_torchrun_cmd(Path("/app/RecStore"), cfg)

        self.assertIn("--dense-arch-layer-sizes", cmd)
        self.assertIn("64,32,16", cmd)
        self.assertIn("--over-arch-layer-sizes", cmd)
        self.assertIn("128,64,1", cmd)

    def test_runner_builds_torchrun_command_with_recstore_runtime_dir(self) -> None:
        runner = RecStoreRunner(Path("/tmp/runtime"))
        cfg = RunConfig(
            backend="recstore",
            nnodes=2,
            node_rank=1,
            nproc_per_node=1,
            master_addr="10.0.2.196",
            master_port=29621,
            rdzv_backend="c10d",
            rdzv_id="recstore-mnmp",
            output_root="/nas/home/shq/docker/rs_demo",
            run_id="recstore-mnmp",
            recstore_runtime_dir="/nas/home/shq/docker/rs_demo/runtime/shared-runtime",
            recstore_main_csv="/nas/home/shq/docker/rs_demo/outputs/recstore-mnmp/recstore_main.csv",
            recstore_main_agg_csv="/nas/home/shq/docker/rs_demo/outputs/recstore-mnmp/recstore_main_agg.csv",
        )

        cmd = runner._build_torchrun_cmd(Path("/app/RecStore"), cfg)

        self.assertIn("--recstore-runtime-dir", cmd)
        self.assertIn(cfg.recstore_runtime_dir, cmd)

    def test_runner_builds_torchrun_command_with_single_node_fast_path_args(self) -> None:
        runner = RecStoreRunner(Path("/tmp/runtime"))
        cfg = RunConfig(
            backend="recstore",
            nnodes=1,
            node_rank=0,
            nproc_per_node=2,
            master_addr="127.0.0.1",
            master_port=29653,
            rdzv_backend="c10d",
            rdzv_id="recstore-fast-path-case",
            output_root="/tmp/rs_demo",
            run_id="recstore-fast-path-case",
            recstore_runtime_dir="/tmp/runtime",
            recstore_main_csv="/tmp/rs_demo/outputs/recstore-fast-path-case/recstore_main.csv",
            recstore_main_agg_csv="/tmp/rs_demo/outputs/recstore-fast-path-case/recstore_main_agg.csv",
            enable_single_node_distributed_fast_path=True,
            single_node_ps_backend="local_shm",
            single_node_owner_policy="hash_mod_world_size",
            read_before_update=False,
            read_mode="direct",
        )

        cmd = runner._build_torchrun_cmd(Path("/app/RecStore"), cfg)

        self.assertIn("--enable-single-node-distributed-fast-path", cmd)
        self.assertIn("--single-node-ps-backend", cmd)
        self.assertIn("local_shm", cmd)
        self.assertIn("--single-node-owner-policy", cmd)
        self.assertIn("hash_mod_world_size", cmd)
        self.assertIn("--no-read-before-update", cmd)

    def test_embedding_module_default_path_does_not_inject_single_node_fast_path(self) -> None:
        cfg = RunConfig(
            backend="recstore",
            steps=1,
            warmup_steps=0,
            init_rows=1,
            batch_size=1,
            embedding_dim=4,
            num_embeddings=16,
            recstore_main_csv="/tmp/recstore-default.csv",
        )

        fake_ebc = self._run_local_worker_with_fake_embedding_module(cfg)

        self.assertFalse(hasattr(fake_ebc, "enable_single_node_distributed_fast_path"))
        self.assertFalse(hasattr(fake_ebc, "single_node_distributed_mode"))
        self.assertFalse(hasattr(fake_ebc, "single_node_ps_backend"))
        self.assertFalse(hasattr(fake_ebc, "single_node_owner_policy"))

    def test_embedding_module_injects_single_node_fast_path_when_enabled(self) -> None:
        cfg = RunConfig(
            backend="recstore",
            steps=1,
            warmup_steps=0,
            init_rows=1,
            batch_size=1,
            embedding_dim=4,
            num_embeddings=16,
            nnodes=1,
            nproc_per_node=2,
            enable_single_node_distributed_fast_path=True,
            single_node_ps_backend="local_shm",
            single_node_owner_policy="hash_mod_world_size",
            recstore_main_csv="/tmp/recstore-fast-path.csv",
        )

        fake_ebc = self._run_local_worker_with_fake_embedding_module(cfg)

        self.assertTrue(fake_ebc.enable_single_node_distributed_fast_path)
        self.assertEqual(fake_ebc.single_node_distributed_mode, "single_node")
        self.assertEqual(fake_ebc.single_node_ps_backend, "local_shm")
        self.assertEqual(fake_ebc.single_node_owner_policy, "hash_mod_world_size")

    def test_gpu_cache_options_are_forwarded_to_recstore_module(self) -> None:
        cfg = RunConfig(
            backend="recstore",
            steps=1,
            warmup_steps=0,
            init_rows=1,
            batch_size=1,
            embedding_dim=4,
            num_embeddings=16,
            enable_gpu_cache=True,
            gpu_cache_capacity=1024,
            recstore_main_csv="/tmp/recstore-gpu-cache.csv",
        )

        fake_ebc = self._run_local_worker_with_fake_embedding_module(cfg)

        self.assertEqual(fake_ebc.kv_client.enable_gpu_cache_calls, [(1024, 4)])

    def test_gpu_cache_lookup_bypass_option_is_forwarded(self) -> None:
        cfg = RunConfig(
            backend="recstore",
            steps=1,
            warmup_steps=0,
            init_rows=1,
            batch_size=1,
            embedding_dim=4,
            num_embeddings=16,
            enable_gpu_cache=True,
            gpu_cache_capacity=1024,
            disable_gpu_cache_lookup_bypass=True,
            recstore_main_csv="/tmp/recstore-gpu-cache-no-bypass.csv",
        )

        fake_ebc = self._run_local_worker_with_fake_embedding_module(cfg)

        self.assertEqual(fake_ebc.kv_client.enable_gpu_cache_calls, [(1024, 4)])
        self.assertFalse(fake_ebc.kv_client.gpu_cache_lookup_bypass_enabled)

    def test_gpu_cache_enable_false_fails_loudly(self) -> None:
        cfg = RunConfig(
            backend="recstore",
            steps=1,
            warmup_steps=0,
            init_rows=1,
            batch_size=1,
            embedding_dim=4,
            num_embeddings=16,
            enable_gpu_cache=True,
            gpu_cache_capacity=1024,
            recstore_main_csv="/tmp/recstore-gpu-cache-false.csv",
        )
        fake_client = _FakeShardedClient()
        fake_client.enable_gpu_cache_result = False
        fake_ebc = types.SimpleNamespace(kv_client=fake_client)

        with self.assertRaisesRegex(RuntimeError, "GPU cache"):
            recstore_runner._configure_gpu_cache(fake_ebc, cfg, embedding_dim=4)

        self.assertEqual(fake_client.enable_gpu_cache_calls, [(1024, 4)])

    def test_gpu_cache_profile_merges_with_stage_prefix(self) -> None:
        row = {}
        fake_client = _FakeShardedClient()
        fake_client.gpu_cache_profile = {
            "gpu_cache_query_ms": 0.11,
            "gpu_cache_backend_lookup_ms": 0.22,
            "gpu_cache_fill_ms": 0.33,
            "gpu_cache_update_ms": 0.44,
            "gpu_cache_hit_count": 5,
        }

        recstore_runner._merge_gpu_cache_profile(row, fake_client, "lookup")

        self.assertEqual(row["lookup_gpu_cache_query_ms"], 0.11)
        self.assertEqual(row["lookup_gpu_cache_backend_lookup_ms"], 0.22)
        self.assertEqual(row["lookup_gpu_cache_fill_ms"], 0.33)
        self.assertEqual(row["lookup_gpu_cache_update_ms"], 0.44)
        self.assertEqual(row["lookup_gpu_cache_hit_count"], 5.0)

    def test_local_worker_switches_client_backend_for_single_node_fast_path(self) -> None:
        cfg = RunConfig(
            backend="recstore",
            steps=1,
            warmup_steps=0,
            init_rows=1,
            batch_size=1,
            embedding_dim=4,
            num_embeddings=16,
            nnodes=1,
            nproc_per_node=2,
            enable_single_node_distributed_fast_path=True,
            single_node_ps_backend="hierkv",
            single_node_owner_policy="hash_mod_world_size",
            recstore_main_csv="/tmp/recstore-hierkv-fast-path.csv",
        )

        runner_runtime = Path(tempfile.mkdtemp())
        repo_root = Path("/app/RecStore")
        dense = torch.zeros((1, 13), dtype=torch.float32)
        sparse = torch.zeros((1, 1), dtype=torch.int64)
        labels = torch.zeros((1, 1), dtype=torch.float32)
        dataset = [(dense, sparse, labels)]
        dataloader = [(dense, sparse, labels)]
        fake_client = _FakeDirectReadShardedClient()
        fake_client_module = types.ModuleType("client")
        fake_client_module.RecstoreClient = lambda library_path=None: object()
        fake_embeddingbag_module = types.ModuleType("python.pytorch.torchrec_kv.EmbeddingBag")
        class _ProfiledEmbeddingBagCollection(_FakeRecStoreEmbeddingBagCollection):
            def __init__(self, *args, **kwargs) -> None:
                super().__init__(*args, **kwargs)
                self._single_node_forward_profile = {
                    "lookup_local_lookup_ms": 1.25,
                    "lookup_wait_ms": 0.75,
                }

        fake_embeddingbag_module.RecStoreEmbeddingBagCollection = _ProfiledEmbeddingBagCollection
        fake_optimizer_module = types.ModuleType("python.pytorch.recstore.optimizer")
        class _ProfiledSparseSGD(_FakeSparseSGD):
            def __init__(self, params, lr: float) -> None:
                super().__init__(params, lr)
                self._last_step_profile = {
                    "exchange_ms": 2.5,
                    "local_update_ms": 3.5,
                    "local_update_backend_call_ms": 3.25,
                    "trace_collect_ms": 0.5,
                    "trace_aggregate_ms": 1.5,
                }

        fake_optimizer_module.SparseSGD = _ProfiledSparseSGD

        _FakeRecStoreEmbeddingBagCollection.last_instance = None

        with ExitStack() as stack:
            stack.enter_context(
                mock.patch.dict(
                    "sys.modules",
                    {
                        "client": fake_client_module,
                        "python.pytorch.torchrec_kv.EmbeddingBag": fake_embeddingbag_module,
                        "python.pytorch.recstore.optimizer": fake_optimizer_module,
                    },
                )
            )
            stack.enter_context(
                mock.patch("model_zoo.rs_demo.runners.recstore_runner.inject_project_paths", lambda *_: None)
            )
            stack.enter_context(
                mock.patch("model_zoo.rs_demo.runners.recstore_runner.torch.manual_seed", lambda *_: None)
            )
            stack.enter_context(
                mock.patch(
                    "model_zoo.rs_demo.runners.recstore_runner.torch.optim.SGD",
                    _FakeDenseOptimizer,
                )
            )
            stack.enter_context(
                mock.patch(
                    "model_zoo.rs_demo.runners.recstore_runner.detect_library_path",
                    lambda *_: repo_root / "build/lib/lib_recstore_ops.so",
                )
            )
            stack.enter_context(
                mock.patch(
                    "model_zoo.rs_demo.runners.recstore_runner.ShardedRecstoreClient",
                    lambda raw_client, runtime_dir: fake_client,
                )
            )
            stack.enter_context(
                mock.patch(
                    "model_zoo.rs_demo.runners.recstore_runner.get_default_cat_names",
                    lambda: ["cat_0"],
                )
            )
            stack.enter_context(
                mock.patch(
                    "model_zoo.rs_demo.runners.recstore_runner.build_train_dataloader",
                    lambda **kwargs: (dataset, dataloader),
                )
            )
            stack.enter_context(
                mock.patch(
                    "model_zoo.rs_demo.runners.recstore_runner.build_kjt_batch_from_dense_sparse_labels",
                    lambda *args, **kwargs: (None, object()),
                )
            )
            stack.enter_context(
                mock.patch(
                    "model_zoo.rs_demo.runners.recstore_runner.build_hybrid_dense_arch",
                    lambda *args, **kwargs: _DummyDense().to(kwargs["device"]),
                )
            )
            stack.enter_context(
                mock.patch(
                    "model_zoo.rs_demo.runners.recstore_runner.reshape_torchrec_embeddings_for_dlrm",
                    lambda **kwargs: torch.zeros((1, 1, 4), dtype=torch.float32, requires_grad=True),
                )
            )
            stack.enter_context(
                mock.patch(
                    "model_zoo.rs_demo.runners.recstore_runner.prepare_hybrid_dlrm_input",
                    lambda **kwargs: (
                        torch.zeros((1, 13), dtype=torch.float32, device=kwargs["device"]),
                        torch.zeros((1, 1, 4), dtype=torch.float32, device=kwargs["device"], requires_grad=True),
                        torch.zeros((1, 1), dtype=torch.float32, device=kwargs["device"]),
                    ),
                )
            )
            stack.enter_context(
                mock.patch(
                    "model_zoo.rs_demo.runners.recstore_runner.finalize_recstore_row",
                    lambda row, **kwargs: row,
                )
            )
            stack.enter_context(
                mock.patch(
                    "model_zoo.rs_demo.runners.recstore_runner.write_stage_csv",
                    lambda *args, **kwargs: None,
                )
            )
            stack.enter_context(
                mock.patch(
                    "model_zoo.rs_demo.runners.recstore_runner.sync_device",
                    lambda *args, **kwargs: None,
                )
            )
            stack.enter_context(
                mock.patch(
                    "model_zoo.rs_demo.runners.recstore_runner.dist",
                    types.SimpleNamespace(
                        is_initialized=lambda: False,
                        init_process_group=lambda **kwargs: None,
                        barrier=lambda *args, **kwargs: None,
                        destroy_process_group=lambda: None,
                    ),
                    create=True,
                )
            )

            runner = RecStoreRunner(runner_runtime)
            runner._run_local_worker(
                repo_root=repo_root,
                cfg=cfg,
                rank=0,
                world_size=1,
                local_rank=0,
                out_csv=runner_runtime / "rank.csv",
            )

        self.assertEqual(fake_client.set_ps_backend_calls, ["hierkv"])
        self.assertEqual(fake_client.activate_shard_calls, [0])

    def test_read_before_update_prefetch_mode_uses_ebc_prefetch_and_sparse_optimizer(self) -> None:
        runner_runtime = Path(tempfile.mkdtemp())
        repo_root = Path("/app/RecStore")
        cfg = RunConfig(
            steps=1,
            warmup_steps=0,
            init_rows=1,
            batch_size=1,
            embedding_dim=4,
            num_embeddings=16,
            read_before_update=True,
            read_mode="prefetch",
            recstore_main_csv=str(runner_runtime / "main.csv"),
        )

        dense = torch.zeros((1, 13), dtype=torch.float32)
        sparse = torch.zeros((1, 1), dtype=torch.int64)
        labels = torch.zeros((1, 1), dtype=torch.float32)
        dataset = [(dense, sparse, labels)]
        dataloader = [(dense, sparse, labels)]

        fake_client = _FakeShardedClient()
        fake_client_module = types.ModuleType("client")
        fake_client_module.RecstoreClient = lambda library_path=None: object()
        fake_embeddingbag_module = types.ModuleType("python.pytorch.torchrec_kv.EmbeddingBag")
        fake_embeddingbag_module.RecStoreEmbeddingBagCollection = _FakeRecStoreEmbeddingBagCollection
        fake_optimizer_module = types.ModuleType("python.pytorch.recstore.optimizer")
        fake_optimizer_module.SparseSGD = _FakeSparseSGD

        with mock.patch.dict(
            "sys.modules",
            {
                "client": fake_client_module,
                "python.pytorch.torchrec_kv.EmbeddingBag": fake_embeddingbag_module,
                "python.pytorch.recstore.optimizer": fake_optimizer_module,
            },
        ):
            with mock.patch("model_zoo.rs_demo.runners.recstore_runner.inject_project_paths", lambda *_: None):
                with mock.patch("model_zoo.rs_demo.runners.recstore_runner.torch.manual_seed", lambda *_: None):
                    with mock.patch(
                        "model_zoo.rs_demo.runners.recstore_runner.torch.optim.SGD",
                        _FakeDenseOptimizer,
                    ):
                        with mock.patch(
                            "model_zoo.rs_demo.runners.recstore_runner.detect_library_path",
                            lambda *_: repo_root / "build/lib/lib_recstore_ops.so",
                        ):
                            with mock.patch(
                                "model_zoo.rs_demo.runners.recstore_runner.ShardedRecstoreClient",
                                lambda raw_client, runtime_dir: fake_client,
                            ):
                                with mock.patch(
                                    "model_zoo.rs_demo.runners.recstore_runner.get_default_cat_names",
                                    lambda: ["cat_0"],
                                ):
                                    with mock.patch(
                                        "model_zoo.rs_demo.runners.recstore_runner.build_train_dataloader",
                                        lambda **kwargs: (dataset, dataloader),
                                    ):
                                        with mock.patch(
                                            "model_zoo.rs_demo.runners.recstore_runner.build_kjt_batch_from_dense_sparse_labels",
                                            lambda *args, **kwargs: (None, object()),
                                        ):
                                            with mock.patch(
                                                "model_zoo.rs_demo.runners.recstore_runner.build_hybrid_dense_arch",
                                                lambda *args, **kwargs: _DummyDense().to(kwargs["device"]),
                                            ):
                                                with mock.patch(
                                                    "model_zoo.rs_demo.runners.recstore_runner.reshape_torchrec_embeddings_for_dlrm",
                                                    lambda **kwargs: torch.zeros((1, 1, 4), dtype=torch.float32, requires_grad=True),
                                                ):
                                                    with mock.patch(
                                                        "model_zoo.rs_demo.runners.recstore_runner.prepare_hybrid_dlrm_input",
                                                        lambda **kwargs: (
                                                            torch.zeros((1, 13), dtype=torch.float32, device=kwargs["device"]),
                                                            torch.zeros((1, 1, 4), dtype=torch.float32, device=kwargs["device"], requires_grad=True),
                                                            torch.zeros((1, 1), dtype=torch.float32, device=kwargs["device"]),
                                                        ),
                                                    ):
                                                        with mock.patch(
                                                            "model_zoo.rs_demo.runners.recstore_runner.run_hybrid_backward",
                                                            lambda **kwargs: torch.zeros((1, 1, 4), dtype=torch.float32),
                                                        ):
                                                            with mock.patch(
                                                                "model_zoo.rs_demo.runners.recstore_runner.sync_device",
                                                                lambda *args, **kwargs: None,
                                                            ):
                                                                with mock.patch(
                                                                    "model_zoo.rs_demo.runners.recstore_runner.finalize_recstore_row",
                                                                    lambda row: row,
                                                                ):
                                                                    with mock.patch(
                                                                        "model_zoo.rs_demo.runners.recstore_runner.summarize_us",
                                                                        lambda xs: "ok",
                                                                    ):
                                                                        with mock.patch(
                                                                            "model_zoo.rs_demo.runners.recstore_runner.write_stage_csv",
                                                                            lambda *args, **kwargs: None,
                                                                        ):
                                                                            runner = RecStoreRunner(runner_runtime)
                                                                            runner.run(repo_root=repo_root, cfg=cfg)

        fake_ebc = _FakeRecStoreEmbeddingBagCollection.last_instance
        fake_sparse_optimizer = _FakeSparseSGD.last_instance
        self.assertIsNotNone(fake_ebc)
        self.assertIsNotNone(fake_sparse_optimizer)
        self.assertEqual(fake_ebc.issue_fused_prefetch_calls, 1)
        self.assertEqual(fake_ebc.issue_fused_prefetch_record_flags, [True])
        self.assertIs(fake_ebc.kwargs["kv_client"], fake_client)
        self.assertEqual(fake_client.emb_read_prefetch_calls, 0)
        self.assertEqual(fake_sparse_optimizer.step_calls, 1)
        self.assertEqual(fake_sparse_optimizer.flush_calls, 1)
        self.assertGreaterEqual(fake_sparse_optimizer.zero_grad_calls, 2)
        self.assertEqual(fake_ebc.reset_perf_stats_calls, 1)
        self.assertEqual(fake_sparse_optimizer.reset_perf_stats_calls, 1)

    def test_read_before_update_prefetch_depth_uses_lookahead_handles(self) -> None:
        runner_runtime = Path(tempfile.mkdtemp())
        repo_root = Path("/app/RecStore")
        cfg = RunConfig(
            steps=3,
            warmup_steps=0,
            init_rows=1,
            batch_size=1,
            embedding_dim=4,
            num_embeddings=16,
            read_before_update=True,
            read_mode="prefetch",
            prefetch_depth=1,
            recstore_main_csv=str(runner_runtime / "main.csv"),
        )

        dense = torch.zeros((1, 13), dtype=torch.float32)
        sparse = torch.zeros((1, 1), dtype=torch.int64)
        labels = torch.zeros((1, 1), dtype=torch.float32)
        dataset = [(dense, sparse, labels)] * 3
        dataloader = list(dataset)

        fake_client = _FakeShardedClient()
        fake_client_module = types.ModuleType("client")
        fake_client_module.RecstoreClient = lambda library_path=None: object()
        fake_embeddingbag_module = types.ModuleType("python.pytorch.torchrec_kv.EmbeddingBag")
        fake_embeddingbag_module.RecStoreEmbeddingBagCollection = _FakeRecStoreEmbeddingBagCollection
        fake_optimizer_module = types.ModuleType("python.pytorch.recstore.optimizer")
        fake_optimizer_module.SparseSGD = _FakeSparseSGD
        captured_rows: list[dict] = []

        with mock.patch.dict(
            "sys.modules",
            {
                "client": fake_client_module,
                "python.pytorch.torchrec_kv.EmbeddingBag": fake_embeddingbag_module,
                "python.pytorch.recstore.optimizer": fake_optimizer_module,
            },
        ):
            with mock.patch("model_zoo.rs_demo.runners.recstore_runner.inject_project_paths", lambda *_: None):
                with mock.patch("model_zoo.rs_demo.runners.recstore_runner.torch.manual_seed", lambda *_: None):
                    with mock.patch(
                        "model_zoo.rs_demo.runners.recstore_runner.torch.optim.SGD",
                        _FakeDenseOptimizer,
                    ):
                        with mock.patch(
                            "model_zoo.rs_demo.runners.recstore_runner.detect_library_path",
                            lambda *_: repo_root / "build/lib/lib_recstore_ops.so",
                        ):
                            with mock.patch(
                                "model_zoo.rs_demo.runners.recstore_runner.ShardedRecstoreClient",
                                lambda raw_client, runtime_dir: fake_client,
                            ):
                                with mock.patch(
                                    "model_zoo.rs_demo.runners.recstore_runner.get_default_cat_names",
                                    lambda: ["cat_0"],
                                ):
                                    with mock.patch(
                                        "model_zoo.rs_demo.runners.recstore_runner.build_train_dataloader",
                                        lambda **kwargs: (dataset, dataloader),
                                    ):
                                        with mock.patch(
                                            "model_zoo.rs_demo.runners.recstore_runner.build_kjt_batch_from_dense_sparse_labels",
                                            lambda *args, **kwargs: (None, object()),
                                        ):
                                            with mock.patch(
                                                "model_zoo.rs_demo.runners.recstore_runner.build_hybrid_dense_arch",
                                                lambda *args, **kwargs: _DummyDense().to(kwargs["device"]),
                                            ):
                                                with mock.patch(
                                                    "model_zoo.rs_demo.runners.recstore_runner.reshape_torchrec_embeddings_for_dlrm",
                                                    lambda **kwargs: torch.zeros((1, 1, 4), dtype=torch.float32, requires_grad=True),
                                                ):
                                                    with mock.patch(
                                                        "model_zoo.rs_demo.runners.recstore_runner.prepare_hybrid_dlrm_input",
                                                        lambda **kwargs: (
                                                            torch.zeros((1, 13), dtype=torch.float32, device=kwargs["device"]),
                                                            torch.zeros((1, 1, 4), dtype=torch.float32, device=kwargs["device"], requires_grad=True),
                                                            torch.zeros((1, 1), dtype=torch.float32, device=kwargs["device"]),
                                                        ),
                                                    ):
                                                        with mock.patch(
                                                            "model_zoo.rs_demo.runners.recstore_runner.run_hybrid_backward",
                                                            lambda **kwargs: torch.zeros((1, 1, 4), dtype=torch.float32),
                                                        ):
                                                            with mock.patch(
                                                                "model_zoo.rs_demo.runners.recstore_runner.sync_device",
                                                                lambda *args, **kwargs: None,
                                                            ):
                                                                with mock.patch(
                                                                    "model_zoo.rs_demo.runners.recstore_runner.finalize_recstore_row",
                                                                    lambda row: row,
                                                                ):
                                                                    with mock.patch(
                                                                        "model_zoo.rs_demo.runners.recstore_runner.summarize_us",
                                                                        lambda xs: "ok",
                                                                    ):
                                                                        with mock.patch(
                                                                            "model_zoo.rs_demo.runners.recstore_runner.write_stage_csv",
                                                                            lambda path, rows: captured_rows.extend(rows),
                                                                        ):
                                                                            runner = RecStoreRunner(runner_runtime)
                                                                            runner.run(repo_root=repo_root, cfg=cfg)

        fake_ebc = _FakeRecStoreEmbeddingBagCollection.last_instance
        self.assertIsNotNone(fake_ebc)
        self.assertEqual(fake_ebc.issue_fused_prefetch_record_flags, [False, False, False])
        self.assertEqual(fake_ebc.set_fused_prefetch_handle_calls, 3)
        self.assertEqual([row["prefetch_depth"] for row in captured_rows], [1.0, 1.0, 1.0])
        self.assertEqual([row["prefetch_consumed_batches"] for row in captured_rows], [1.0, 1.0, 1.0])
        self.assertEqual([row["prefetch_consumed_total_ids"] for row in captured_rows], [7.0, 7.0, 7.0])
        self.assertTrue(all(row["prefetch_issue_to_consume_ms"] >= 0 for row in captured_rows))

    def test_local_worker_emits_perf_breakdown_columns_from_model_layer_stats(self) -> None:
        runner_runtime = Path(tempfile.mkdtemp())
        repo_root = Path("/app/RecStore")
        cfg = RunConfig(
            backend="recstore",
            steps=1,
            warmup_steps=0,
            init_rows=1,
            batch_size=1,
            embedding_dim=4,
            num_embeddings=16,
            read_before_update=True,
            read_mode="prefetch",
            recstore_main_csv=str(runner_runtime / "main.csv"),
        )

        dense = torch.zeros((1, 13), dtype=torch.float32)
        sparse = torch.zeros((1, 1), dtype=torch.int64)
        labels = torch.zeros((1, 1), dtype=torch.float32)
        dataset = [(dense, sparse, labels)]
        dataloader = [(dense, sparse, labels)]

        fake_client = _FakeShardedClient()
        fake_client_module = types.ModuleType("client")
        fake_client_module.RecstoreClient = lambda library_path=None: object()
        fake_embeddingbag_module = types.ModuleType("python.pytorch.torchrec_kv.EmbeddingBag")
        fake_embeddingbag_module.RecStoreEmbeddingBagCollection = _FakeRecStoreEmbeddingBagCollection
        fake_optimizer_module = types.ModuleType("python.pytorch.recstore.optimizer")
        fake_optimizer_module.SparseSGD = _FakeSparseSGD
        captured_rows = []

        with mock.patch.dict(
            "sys.modules",
            {
                "client": fake_client_module,
                "python.pytorch.torchrec_kv.EmbeddingBag": fake_embeddingbag_module,
                "python.pytorch.recstore.optimizer": fake_optimizer_module,
            },
        ):
            with mock.patch("model_zoo.rs_demo.runners.recstore_runner.inject_project_paths", lambda *_: None):
                with mock.patch("model_zoo.rs_demo.runners.recstore_runner.torch.manual_seed", lambda *_: None):
                    with mock.patch(
                        "model_zoo.rs_demo.runners.recstore_runner.torch.optim.SGD",
                        _FakeDenseOptimizer,
                    ):
                        with mock.patch(
                            "model_zoo.rs_demo.runners.recstore_runner.detect_library_path",
                            lambda *_: repo_root / "build/lib/lib_recstore_ops.so",
                        ):
                            with mock.patch(
                                "model_zoo.rs_demo.runners.recstore_runner.ShardedRecstoreClient",
                                lambda raw_client, runtime_dir: fake_client,
                            ):
                                with mock.patch(
                                    "model_zoo.rs_demo.runners.recstore_runner.get_default_cat_names",
                                    lambda: ["cat_0"],
                                ):
                                    with mock.patch(
                                        "model_zoo.rs_demo.runners.recstore_runner.build_train_dataloader",
                                        lambda **kwargs: (dataset, dataloader),
                                    ):
                                        with mock.patch(
                                            "model_zoo.rs_demo.runners.recstore_runner.build_kjt_batch_from_dense_sparse_labels",
                                            lambda *args, **kwargs: (None, object()),
                                        ):
                                            with mock.patch(
                                                "model_zoo.rs_demo.runners.recstore_runner.build_hybrid_dense_arch",
                                                lambda *args, **kwargs: _DummyDense().to(kwargs["device"]),
                                            ):
                                                with mock.patch(
                                                    "model_zoo.rs_demo.runners.recstore_runner.reshape_torchrec_embeddings_for_dlrm",
                                                    lambda **kwargs: torch.zeros((1, 1, 4), dtype=torch.float32, requires_grad=True),
                                                ):
                                                    with mock.patch(
                                                        "model_zoo.rs_demo.runners.recstore_runner.prepare_hybrid_dlrm_input",
                                                        lambda **kwargs: (
                                                            torch.zeros((1, 13), dtype=torch.float32, device=kwargs["device"]),
                                                            torch.zeros((1, 1, 4), dtype=torch.float32, device=kwargs["device"], requires_grad=True),
                                                            torch.zeros((1, 1), dtype=torch.float32, device=kwargs["device"]),
                                                        ),
                                                    ):
                                                        with mock.patch(
                                                            "model_zoo.rs_demo.runners.recstore_runner.run_hybrid_backward",
                                                            lambda **kwargs: torch.zeros((1, 1, 4), dtype=torch.float32),
                                                        ):
                                                            with mock.patch(
                                                                "model_zoo.rs_demo.runners.recstore_runner.sync_device",
                                                                lambda *args, **kwargs: None,
                                                            ):
                                                                with mock.patch(
                                                                    "model_zoo.rs_demo.runners.recstore_runner.finalize_recstore_row",
                                                                    lambda row: captured_rows.append(dict(row)) or row,
                                                                ):
                                                                    with mock.patch(
                                                                        "model_zoo.rs_demo.runners.recstore_runner.summarize_us",
                                                                        lambda xs: "ok",
                                                                    ):
                                                                        with mock.patch(
                                                                            "model_zoo.rs_demo.runners.recstore_runner.write_stage_csv",
                                                                            lambda *args, **kwargs: None,
                                                                        ):
                                                                            runner = RecStoreRunner(runner_runtime)
                                                                            runner.run(repo_root=repo_root, cfg=cfg)

        self.assertEqual(len(captured_rows), 1)
        row = captured_rows[0]
        self.assertEqual(row["prefetch_issue_ms"], 0.2)
        self.assertEqual(row["lookup_wait_ms"], 0.6)
        self.assertEqual(row["lookup_owner_exchange_ms"], 0.4)
        self.assertEqual(row["lookup_local_lookup_ms"], 0.5)
        self.assertEqual(row["lookup_reassemble_ms"], 0.3)
        self.assertEqual(row["pool_embedding_bag_ms"], 0.7)
        self.assertEqual(row["update_trace_merge_ms"], 0.25)
        self.assertEqual(row["update_owner_exchange_ms"], 0.35)
        self.assertEqual(row["update_local_apply_ms"], 0.45)
        self.assertEqual(row["update_async_enqueue_ms"], 0.05)
        self.assertEqual(row["update_flush_wait_ms"], 0.15)

    def test_runner_exports_fast_path_profiles_into_rows(self) -> None:
        runner_runtime = Path(tempfile.mkdtemp())
        repo_root = Path("/app/RecStore")
        cfg = RunConfig(
            steps=1,
            warmup_steps=0,
            init_rows=1,
            batch_size=1,
            embedding_dim=4,
            num_embeddings=16,
            read_before_update=False,
            recstore_main_csv=str(runner_runtime / "main.csv"),
        )

        dense = torch.zeros((1, 13), dtype=torch.float32)
        sparse = torch.zeros((1, 1), dtype=torch.int64)
        labels = torch.zeros((1, 1), dtype=torch.float32)
        dataset = [(dense, sparse, labels)]
        dataloader = [(dense, sparse, labels)]

        fake_client = _FakeDirectReadShardedClient()
        fake_client_module = types.ModuleType("client")
        fake_client_module.RecstoreClient = lambda library_path=None: object()
        fake_embeddingbag_module = types.ModuleType("python.pytorch.torchrec_kv.EmbeddingBag")
        class _ProfiledEmbeddingBagCollection(_FakeRecStoreEmbeddingBagCollection):
            def __init__(self, *args, **kwargs) -> None:
                super().__init__(*args, **kwargs)
                self._single_node_forward_profile = {
                    "lookup_local_lookup_ms": 1.25,
                    "lookup_wait_ms": 0.75,
                }

        fake_embeddingbag_module.RecStoreEmbeddingBagCollection = _ProfiledEmbeddingBagCollection
        fake_optimizer_module = types.ModuleType("python.pytorch.recstore.optimizer")
        class _ProfiledSparseSGD(_FakeSparseSGD):
            def __init__(self, params, lr: float) -> None:
                super().__init__(params, lr)
                self._last_step_profile = {
                    "exchange_ms": 2.5,
                    "local_update_ms": 3.5,
                    "local_update_backend_call_ms": 3.25,
                    "trace_collect_ms": 0.5,
                    "trace_aggregate_ms": 1.5,
                }

        fake_optimizer_module.SparseSGD = _ProfiledSparseSGD

        with mock.patch.dict(
            "sys.modules",
            {
                "client": fake_client_module,
                "python.pytorch.torchrec_kv.EmbeddingBag": fake_embeddingbag_module,
                "python.pytorch.recstore.optimizer": fake_optimizer_module,
            },
        ):
            with mock.patch("model_zoo.rs_demo.runners.recstore_runner.inject_project_paths", lambda *_: None):
                with mock.patch(
                    "model_zoo.rs_demo.runners.recstore_runner.detect_library_path",
                    lambda *_: repo_root / "build/lib/lib_recstore_ops.so",
                ):
                    with mock.patch(
                        "model_zoo.rs_demo.runners.recstore_runner.ShardedRecstoreClient",
                        lambda raw_client, runtime_dir: fake_client,
                    ):
                        with mock.patch(
                            "model_zoo.rs_demo.runners.recstore_runner.get_default_cat_names",
                            lambda: ["cat_0"],
                        ):
                            with mock.patch(
                                "model_zoo.rs_demo.runners.recstore_runner.build_train_dataloader",
                                lambda **kwargs: (dataset, dataloader),
                            ):
                                with mock.patch(
                                    "model_zoo.rs_demo.runners.recstore_runner.build_kjt_batch_from_dense_sparse_labels",
                                    lambda *args, **kwargs: (None, object()),
                                ):
                                    with mock.patch(
                                        "model_zoo.rs_demo.runners.recstore_runner.build_hybrid_dense_arch",
                                        lambda *args, **kwargs: _DummyDense().to(kwargs["device"]),
                                    ):
                                        with mock.patch(
                                            "model_zoo.rs_demo.runners.recstore_runner.reshape_torchrec_embeddings_for_dlrm",
                                            lambda **kwargs: torch.zeros((1, 1, 4), dtype=torch.float32, requires_grad=True),
                                        ):
                                            with mock.patch(
                                                "model_zoo.rs_demo.runners.recstore_runner.prepare_hybrid_dlrm_input",
                                                lambda **kwargs: (
                                                    torch.zeros((1, 13), dtype=torch.float32, device=kwargs["device"]),
                                                    torch.zeros((1, 1, 4), dtype=torch.float32, device=kwargs["device"], requires_grad=True),
                                                    torch.zeros((1, 1), dtype=torch.float32, device=kwargs["device"]),
                                                ),
                                            ):
                                                with mock.patch(
                                                    "model_zoo.rs_demo.runners.recstore_runner.run_hybrid_backward",
                                                    lambda **kwargs: torch.zeros((1, 1, 4), dtype=torch.float32),
                                                ):
                                                    with mock.patch(
                                                        "model_zoo.rs_demo.runners.recstore_runner.sync_device",
                                                        lambda *args, **kwargs: None,
                                                    ):
                                                        with mock.patch(
                                                            "model_zoo.rs_demo.runners.recstore_runner.summarize_us",
                                                            lambda xs: "ok",
                                                        ):
                                                            with mock.patch(
                                                                "model_zoo.rs_demo.runners.recstore_runner.torch.manual_seed",
                                                                lambda *_: None,
                                                            ):
                                                                runner = RecStoreRunner(runner_runtime)
                                                                runner.run(repo_root=repo_root, cfg=cfg)

        with Path(cfg.recstore_main_csv).open("r", encoding="utf-8") as f:
            rows = list(csv.DictReader(f))

        self.assertEqual(len(rows), 1)
        self.assertEqual(float(rows[0]["lookup_local_lookup_ms"]), 1.25)
        self.assertEqual(float(rows[0]["lookup_wait_ms"]), 0.75)
        self.assertEqual(float(rows[0]["exchange_ms"]), 2.5)
        self.assertEqual(float(rows[0]["local_update_ms"]), 3.5)
        self.assertEqual(float(rows[0]["local_update_backend_call_ms"]), 3.25)
        self.assertEqual(float(rows[0]["trace_collect_ms"]), 0.5)
        self.assertEqual(float(rows[0]["trace_aggregate_ms"]), 1.5)
        self.assertIn("sparse_backward_replay_ms", rows[0])
        self.assertIn("sparse_optimizer_step_ms", rows[0])
        self.assertIn("sparse_optimizer_flush_ms", rows[0])
        self.assertIn("sparse_zero_grad_ms", rows[0])

    def test_runner_passes_compute_device_into_sparse_feature_builder(self) -> None:
        runner_runtime = Path(tempfile.mkdtemp())
        repo_root = Path("/app/RecStore")
        cfg = RunConfig(
            steps=1,
            warmup_steps=0,
            init_rows=1,
            batch_size=1,
            embedding_dim=4,
            num_embeddings=16,
            read_before_update=False,
            recstore_main_csv=str(runner_runtime / "main.csv"),
        )

        dense = torch.zeros((1, 13), dtype=torch.float32)
        sparse = torch.zeros((1, 1), dtype=torch.int64)
        labels = torch.zeros((1, 1), dtype=torch.float32)
        dataset = [(dense, sparse, labels)]
        dataloader = [(dense, sparse, labels)]

        fake_client = _FakeDirectReadShardedClient()
        fake_client_module = types.ModuleType("client")
        fake_client_module.RecstoreClient = lambda library_path=None: object()
        fake_embeddingbag_module = types.ModuleType("python.pytorch.torchrec_kv.EmbeddingBag")
        fake_embeddingbag_module.RecStoreEmbeddingBagCollection = _FakeRecStoreEmbeddingBagCollection
        fake_optimizer_module = types.ModuleType("python.pytorch.recstore.optimizer")
        fake_optimizer_module.SparseSGD = _FakeSparseSGD
        device_calls: list[torch.device] = []

        def _build_sparse_features(*args, **kwargs):
            device_calls.append(kwargs["device"])
            return None, object()

        with mock.patch.dict(
            "sys.modules",
            {
                "client": fake_client_module,
                "python.pytorch.torchrec_kv.EmbeddingBag": fake_embeddingbag_module,
                "python.pytorch.recstore.optimizer": fake_optimizer_module,
            },
        ):
            with mock.patch("model_zoo.rs_demo.runners.recstore_runner.inject_project_paths", lambda *_: None):
                with mock.patch(
                    "model_zoo.rs_demo.runners.recstore_runner.detect_library_path",
                    lambda *_: repo_root / "build/lib/lib_recstore_ops.so",
                ):
                    with mock.patch(
                        "model_zoo.rs_demo.runners.recstore_runner.ShardedRecstoreClient",
                        lambda raw_client, runtime_dir: fake_client,
                    ):
                        with mock.patch(
                            "model_zoo.rs_demo.runners.recstore_runner.get_default_cat_names",
                            lambda: ["cat_0"],
                        ):
                            with mock.patch(
                                "model_zoo.rs_demo.runners.recstore_runner.build_train_dataloader",
                                lambda **kwargs: (dataset, dataloader),
                            ):
                                with mock.patch(
                                    "model_zoo.rs_demo.runners.recstore_runner.build_kjt_batch_from_dense_sparse_labels",
                                    _build_sparse_features,
                                ):
                                    with mock.patch(
                                        "model_zoo.rs_demo.runners.recstore_runner.build_hybrid_dense_arch",
                                        lambda *args, **kwargs: _DummyDense().to(kwargs["device"]),
                                    ):
                                        with mock.patch(
                                            "model_zoo.rs_demo.runners.recstore_runner.reshape_torchrec_embeddings_for_dlrm",
                                            lambda **kwargs: torch.zeros((1, 1, 4), dtype=torch.float32, requires_grad=True),
                                        ):
                                            with mock.patch(
                                                "model_zoo.rs_demo.runners.recstore_runner.prepare_hybrid_dlrm_input",
                                                lambda **kwargs: (
                                                    torch.zeros((1, 13), dtype=torch.float32, device=kwargs["device"]),
                                                    torch.zeros((1, 1, 4), dtype=torch.float32, device=kwargs["device"], requires_grad=True),
                                                    torch.zeros((1, 1), dtype=torch.float32, device=kwargs["device"]),
                                                ),
                                            ):
                                                with mock.patch(
                                                    "model_zoo.rs_demo.runners.recstore_runner.run_hybrid_backward",
                                                    lambda **kwargs: torch.zeros((1, 1, 4), dtype=torch.float32),
                                                ):
                                                    with mock.patch(
                                                        "model_zoo.rs_demo.runners.recstore_runner.sync_device",
                                                        lambda *args, **kwargs: None,
                                                    ):
                                                        with mock.patch(
                                                            "model_zoo.rs_demo.runners.recstore_runner.summarize_us",
                                                            lambda xs: "ok",
                                                        ):
                                                            with mock.patch(
                                                                "model_zoo.rs_demo.runners.recstore_runner.torch.manual_seed",
                                                                lambda *_: None,
                                                            ):
                                                                with mock.patch(
                                                                    "model_zoo.rs_demo.runners.recstore_runner.torch.optim.SGD",
                                                                    _FakeDenseOptimizer,
                                                                ):
                                                                    runner = RecStoreRunner(runner_runtime)
                                                                    runner.run(repo_root=repo_root, cfg=cfg)

        self.assertEqual(len(device_calls), 1)
        expected_device_type = "cuda" if torch.cuda.is_available() else "cpu"
        self.assertEqual(device_calls[0].type, expected_device_type)

    def test_nonzero_rank_skips_table_init_and_warm_write(self) -> None:
        runner_runtime = Path(tempfile.mkdtemp())
        repo_root = Path("/app/RecStore")
        cfg = RunConfig(
            backend="recstore",
            steps=1,
            warmup_steps=0,
            init_rows=1,
            batch_size=1,
            embedding_dim=4,
            num_embeddings=16,
            read_before_update=False,
            nnodes=1,
            nproc_per_node=2,
            output_root=str(runner_runtime),
            recstore_main_csv=str(runner_runtime / "main.csv"),
        )

        dense = torch.zeros((1, 13), dtype=torch.float32)
        sparse = torch.zeros((1, 1), dtype=torch.int64)
        labels = torch.zeros((1, 1), dtype=torch.float32)
        dataset = [(dense, sparse, labels)]
        dataloader = [(dense, sparse, labels)]

        fake_client = _FakeDirectReadShardedClient()
        fake_client_module = types.ModuleType("client")
        fake_client_module.RecstoreClient = lambda library_path=None: object()
        fake_embeddingbag_module = types.ModuleType("python.pytorch.torchrec_kv.EmbeddingBag")
        fake_embeddingbag_module.RecStoreEmbeddingBagCollection = _FakeRecStoreEmbeddingBagCollection
        fake_optimizer_module = types.ModuleType("python.pytorch.recstore.optimizer")
        fake_optimizer_module.SparseSGD = _FakeSparseSGD
        fake_dist = types.SimpleNamespace(
            is_initialized=lambda: False,
            init_process_group=lambda **kwargs: None,
            barrier=lambda *args, **kwargs: None,
            destroy_process_group=lambda: None,
        )

        with ExitStack() as stack:
            stack.enter_context(
                mock.patch.dict(
                    "sys.modules",
                    {
                        "client": fake_client_module,
                        "python.pytorch.torchrec_kv.EmbeddingBag": fake_embeddingbag_module,
                        "python.pytorch.recstore.optimizer": fake_optimizer_module,
                    },
                )
            )
            stack.enter_context(
                mock.patch("model_zoo.rs_demo.runners.recstore_runner.inject_project_paths", lambda *_: None)
            )
            stack.enter_context(
                mock.patch("model_zoo.rs_demo.runners.recstore_runner.torch.manual_seed", lambda *_: None)
            )
            stack.enter_context(
                mock.patch(
                    "model_zoo.rs_demo.runners.recstore_runner.torch.optim.SGD",
                    _FakeDenseOptimizer,
                )
            )
            stack.enter_context(
                mock.patch(
                    "model_zoo.rs_demo.runners.recstore_runner.detect_library_path",
                    lambda *_: repo_root / "build/lib/lib_recstore_ops.so",
                )
            )
            stack.enter_context(
                mock.patch(
                    "model_zoo.rs_demo.runners.recstore_runner.ShardedRecstoreClient",
                    lambda raw_client, runtime_dir: fake_client,
                )
            )
            stack.enter_context(
                mock.patch(
                    "model_zoo.rs_demo.runners.recstore_runner.get_default_cat_names",
                    lambda: ["cat_0"],
                )
            )
            stack.enter_context(
                mock.patch(
                    "model_zoo.rs_demo.runners.recstore_runner.build_train_dataloader",
                    lambda **kwargs: (dataset, dataloader),
                )
            )
            stack.enter_context(
                mock.patch(
                    "model_zoo.rs_demo.runners.recstore_runner.build_kjt_batch_from_dense_sparse_labels",
                    lambda *args, **kwargs: (None, object()),
                )
            )
            stack.enter_context(
                mock.patch(
                    "model_zoo.rs_demo.runners.recstore_runner.build_hybrid_dense_arch",
                    lambda *args, **kwargs: _DummyDense().to(kwargs["device"]),
                )
            )
            stack.enter_context(
                mock.patch(
                    "model_zoo.rs_demo.runners.recstore_runner._maybe_wrap_dense_module_for_dist",
                    lambda **kwargs: kwargs["dense_module"],
                )
            )
            stack.enter_context(
                mock.patch(
                    "model_zoo.rs_demo.runners.recstore_runner.reshape_torchrec_embeddings_for_dlrm",
                    lambda **kwargs: torch.zeros((1, 1, 4), dtype=torch.float32, requires_grad=True),
                )
            )
            stack.enter_context(
                mock.patch(
                    "model_zoo.rs_demo.runners.recstore_runner.prepare_hybrid_dlrm_input",
                    lambda **kwargs: (
                        torch.zeros((1, 13), dtype=torch.float32, device=kwargs["device"]),
                        torch.zeros((1, 1, 4), dtype=torch.float32, device=kwargs["device"], requires_grad=True),
                        torch.zeros((1, 1), dtype=torch.float32, device=kwargs["device"]),
                    ),
                )
            )
            stack.enter_context(
                mock.patch(
                    "model_zoo.rs_demo.runners.recstore_runner.run_hybrid_backward",
                    lambda **kwargs: torch.zeros((1, 1, 4), dtype=torch.float32),
                )
            )
            stack.enter_context(
                mock.patch(
                    "model_zoo.rs_demo.runners.recstore_runner.sync_device",
                    lambda *args, **kwargs: None,
                )
            )
            stack.enter_context(
                mock.patch(
                    "model_zoo.rs_demo.runners.recstore_runner.finalize_recstore_row",
                    lambda row: row,
                )
            )
            stack.enter_context(
                mock.patch(
                    "model_zoo.rs_demo.runners.recstore_runner.summarize_us",
                    lambda xs: "ok",
                )
            )
            stack.enter_context(
                mock.patch(
                    "model_zoo.rs_demo.runners.recstore_runner.write_stage_csv",
                    lambda *args, **kwargs: None,
                )
            )
            stack.enter_context(mock.patch("torch.distributed.is_initialized", fake_dist.is_initialized))
            stack.enter_context(
                mock.patch("torch.distributed.init_process_group", fake_dist.init_process_group)
            )
            stack.enter_context(mock.patch("torch.distributed.barrier", fake_dist.barrier))
            stack.enter_context(
                mock.patch("torch.distributed.destroy_process_group", fake_dist.destroy_process_group)
            )

            runner = RecStoreRunner(runner_runtime)
            runner._run_local_worker(
                repo_root=repo_root,
                cfg=cfg,
                rank=1,
                world_size=2,
                local_rank=0,
                out_csv=runner_runtime / "rank1.csv",
            )

        fake_ebc = _FakeRecStoreEmbeddingBagCollection.last_instance
        self.assertIsNotNone(fake_ebc)
        self.assertFalse(fake_ebc.kwargs["initialize_tables"])
        self.assertEqual(fake_client.emb_write_calls, 0)

    def test_merge_rank_outputs_preserves_rank_order(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            rank0 = Path(tmpdir) / "rank0.csv"
            rank1 = Path(tmpdir) / "rank1.csv"
            out_csv = Path(tmpdir) / "main.csv"
            recstore_runner.write_stage_csv(
                rank1,
                [
                    {
                        "backend": "recstore",
                        "dist_mode": "single_node",
                        "rank": 1,
                        "step": 0,
                        "step_total_ms": 11.0,
                    }
                ],
            )
            recstore_runner.write_stage_csv(
                rank0,
                [
                    {
                        "backend": "recstore",
                        "dist_mode": "single_node",
                        "rank": 0,
                        "step": 1,
                        "step_total_ms": 9.0,
                    },
                    {
                        "backend": "recstore",
                        "dist_mode": "single_node",
                        "rank": 0,
                        "step": 0,
                        "step_total_ms": 10.0,
                    },
                ],
            )

            rows = recstore_runner._merge_rank_outputs([rank1, rank0], out_csv)

            self.assertEqual([(row["rank"], row["step"]) for row in rows], [(0, 0), (0, 1), (1, 0)])
            self.assertTrue(out_csv.exists())

    def test_write_or_verify_worker_fingerprint_rejects_mismatch(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "fingerprints.json"
            recstore_runner._write_or_verify_worker_fingerprint(
                rank=0,
                world_size=2,
                fingerprint={"files": {"a.py": "111"}},
                fingerprint_path=path,
            )
            with self.assertRaisesRegex(RuntimeError, "worker fingerprint mismatch"):
                recstore_runner._write_or_verify_worker_fingerprint(
                    rank=1,
                    world_size=2,
                    fingerprint={"files": {"a.py": "222"}},
                    fingerprint_path=path,
                )

if __name__ == "__main__":
    unittest.main()
