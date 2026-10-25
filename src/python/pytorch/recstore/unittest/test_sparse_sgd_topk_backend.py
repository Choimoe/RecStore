"""Opt-in native PS readback test; use a dedicated test server/configuration.

Set RECSTORE_TOPK_NATIVE_TEST=1 and RECSTORE_CONFIG to an SGD configuration
for a running GRPC or shared LOCAL_SHM server. Run once per transport in a
fresh process. The test writes only a new, uniquely named eight-row table.
"""

import os
from types import SimpleNamespace
import uuid

import pytest
import torch

from recstore import load_ops_library
from recstore.KVClient import RecStoreClient
from recstore.optimizer import SparseSGD, _server_sparse_optimizer_config


@pytest.mark.skipif(
    os.environ.get("RECSTORE_TOPK_NATIVE_TEST") != "1",
    reason="requires an explicitly configured native test PS",
)
@pytest.mark.parametrize("keep_ratio", [0.5, 1.0])
def test_topk_native_update_readback(keep_ratio):
    assert os.environ.get("RECSTORE_CONFIG"), "Set a dedicated RECSTORE_CONFIG"
    config = _server_sparse_optimizer_config()
    assert config["type"] == "SGD"
    assert config["learning_rate"] > 0
    load_ops_library()
    client = RecStoreClient()
    backend = client.current_ps_backend()
    assert backend in ("grpc", "distributed_grpc", "local_shm")
    name = "topk_readback_" + uuid.uuid4().hex
    initial = torch.arange(16, dtype=torch.float32).reshape(8, 2)
    client.init_data(
        name=name,
        shape=(8, 2),
        dtype=torch.float32,
        init_func=lambda shape, dtype: initial.clone(),
    )
    ids = torch.arange(8)
    torch.testing.assert_close(client.pull(name, ids), initial)
    module = SimpleNamespace(
        kv_client=client,
        _config_names={"feature": name},
        _enable_fusion=True,
        _master_config=SimpleNamespace(name=name),
        fast_path_mode="auto" if backend == "local_shm" else "off",
        resolve_fast_path_backend=lambda: None,
        _trace=[(
            name,
            torch.tensor([0, 1, 0, 2]),
            torch.tensor([[1.0, 2.0], [0.1, 0.0], [5.0, 6.0], [0.0, 2.0]]),
        )],
    )
    module.reset_trace = module._trace.clear
    optimizer = SparseSGD([module], lr=config["learning_rate"], keep_ratio=keep_ratio)
    optimizer.step()
    if backend != "local_shm":
        torch.testing.assert_close(client.pull(name, ids), initial)
    optimizer.flush()

    expected = initial.clone()
    expected[0] -= config["learning_rate"] * torch.tensor([6.0, 8.0])
    expected[2] -= config["learning_rate"] * torch.tensor([0.0, 2.0])
    if keep_ratio == 1.0:
        expected[1] -= config["learning_rate"] * torch.tensor([0.1, 0.0])
    torch.testing.assert_close(client.pull(name, ids), expected)
    optimizer.flush()
    torch.testing.assert_close(client.pull(name, ids), expected)
