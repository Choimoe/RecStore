"""Exact AdamW contracts, including checkpointed slots and Quanta rounding."""

from pathlib import Path
import struct

import numpy as np
import pytest

import _recstore_kv


_STATE_HEADER = struct.Struct("=8sIQqqqQQQffq")


def _engine(path, dim, capacity=512, threads=1):
    return _recstore_kv.LocalIndexedKV(
        capacity=capacity,
        embedding_dim=dim,
        index_type="DRAM_PET_HASH",
        value_path=str(path),
        allocator_capacity=4 << 20,
        num_threads=threads,
    )


def _state(engine, path):
    assert engine.save_checkpoint(str(path), "adam-exact")
    files = list(path.parent.glob(path.name + ".*.state"))
    assert len(files) == 1
    payload = files[0].read_bytes()
    (magic, version, metadata_size, capacity, dim, next_index, active_rows,
     free_count, step, beta1, beta2, slot_rows) = _STATE_HEADER.unpack_from(payload)
    assert magic == b"RSIDXS01" and version == 1
    offset = _STATE_HEADER.size + metadata_size + free_count * 8
    count = slot_rows * dim
    assert len(payload) == offset + count * 8 + 8
    m = np.frombuffer(payload, dtype=np.float32, count=count, offset=offset)
    v = np.frombuffer(payload, dtype=np.float32, count=count,
                      offset=offset + count * 4)
    return payload, {
        "m": m.reshape(slot_rows, dim),
        "v": v.reshape(slot_rows, dim),
        "beta1": np.float32(beta1),
        "beta2": np.float32(beta2),
        "step": step,
    }


def _values(engine, ids, dim):
    values = np.empty((ids.size, dim), dtype=np.float32)
    missing = engine.lookup_values_into_raw(
        ids.ctypes.data, ids.size, values.ctypes.data
    )
    assert missing == 0
    return values


def _assert_bits(actual, expected):
    np.testing.assert_array_equal(
        np.asarray(actual, dtype=np.float32).view(np.uint32),
        np.asarray(expected, dtype=np.float32).view(np.uint32),
    )


def _quanta_step(values, m, v, beta1_t, beta2_t, grads, params):
    """Follow block_apply_adamw_op.cc expression types and parentheses."""
    lr, beta1, beta2, eps, weight_decay = params
    beta1_t = np.float32(np.float64(beta1_t) * beta1)
    beta2_t = np.float32(np.float64(beta2_t) * beta2)
    correction1 = np.float32(1.0) - beta1_t
    correction2 = np.float32(1.0) - beta2_t
    alpha = np.float32(
        -np.float64(lr) / np.float64(correction1)
        * np.float64(np.sqrt(correction2))
    )
    grad64 = grads.astype(np.float64)
    values = (values.astype(np.float64) * (1.0 - weight_decay)).astype(np.float32)
    m = (m.astype(np.float64) * beta1 + grad64 * (1.0 - beta1)).astype(np.float32)
    v = (v.astype(np.float64) * beta2
         + ((1.0 - beta2) * grad64) * grad64).astype(np.float32)
    values = (
        values.astype(np.float64)
        + np.float64(alpha)
        * (m.astype(np.float64) / (np.sqrt(v).astype(np.float64) + eps))
    ).astype(np.float32)
    return values, m, v, beta1_t, beta2_t


@pytest.mark.parametrize("dim", [1, 3, 4, 5, 7, 16, 17, 33])
@pytest.mark.parametrize("prefetch", [0, 4])
def test_scalar_simd_and_quanta_rounding_exact(tmp_path, dim, prefetch):
    rng = np.random.default_rng(9817 + dim)
    rows = 37
    ids = rng.choice(10000, size=rows, replace=False).astype(np.int64)
    indices = np.arange(rows, dtype=np.int64) * 2 + 1
    initial = rng.normal(0, 0.3, (rows, dim)).astype(np.float32)
    scalar = _engine(tmp_path / "scalar", dim)
    vector = _engine(tmp_path / "vector", dim)
    for engine in (scalar, vector):
        engine.bulk_insert_indexed(ids, indices, initial)
    scalar_config = scalar.set_adam_execution(False, prefetch)
    vector_config = vector.set_adam_execution(True, prefetch)
    assert not scalar_config["vectorized_active"]
    assert vector_config["vectorized_active"] == (
        vector_config["avx2_available"] and dim >= 4
    )
    expected = initial.copy()
    m = np.zeros_like(initial)
    v = np.zeros_like(initial)
    beta1_t = beta2_t = np.float32(1)
    parameters = [
        (0.003, 0.8, 0.95, 1e-6, 0.02),
        (0.0007, 0.9, 0.999, 1e-8, 0.001),
        (0.013, 0.123456789, 0.987654321, 1e-12, 0.13),
        (0.0, 0.0, 0.0, 1e-5, 0.0),
        (1e-9, 0.99, 0.9999, 0.0, 0.0),
        (0.07, 0.83, 0.92, 0.03, 1.2),
    ]
    for step in range(12):
        params = parameters[step % len(parameters)]
        grads = rng.normal(0, 0.7, initial.shape).astype(np.float32)
        # Include very small, large, and signed-zero inputs with finite slots.
        grads.flat[0] = np.float32(1e-20)
        grads.flat[1] = np.float32(-1e6)
        grads.flat[2] = np.float32(-0.0 if params[3] != 0.0 else 0.5)
        order = rng.permutation(rows)
        query = np.ascontiguousarray(ids[order])
        update = np.ascontiguousarray(grads[order])
        expected, m, v, beta1_t, beta2_t = _quanta_step(
            expected, m, v, beta1_t, beta2_t, grads, params
        )
        for engine in (scalar, vector):
            profile = engine.adamw_tf_update_indexed(query, update, *params)
            assert profile["adam_prefetch_distance"] == prefetch
        _assert_bits(_values(scalar, ids, dim), expected)
        _assert_bits(_values(vector, ids, dim), expected)
        scalar_bytes, state = _state(scalar, tmp_path / "scalar.checkpoint")
        vector_bytes, _ = _state(vector, tmp_path / "vector.checkpoint")
        assert scalar_bytes == vector_bytes
        _assert_bits(state["m"][indices], m)
        _assert_bits(state["v"][indices], v)
        _assert_bits(state["beta1"], beta1_t)
        _assert_bits(state["beta2"], beta2_t)
        assert state["step"] == step + 1


def test_parallel_simd_matches_scalar(tmp_path):
    rng = np.random.default_rng(593)
    rows, dim = 8193, 7
    ids = np.arange(rows, dtype=np.int64) + 100
    indices = np.arange(rows, dtype=np.int64)
    values = rng.normal(size=(rows, dim)).astype(np.float32)
    scalar = _engine(tmp_path / "scalar", dim, capacity=16384, threads=4)
    vector = _engine(tmp_path / "vector", dim, capacity=16384, threads=4)
    for engine in (scalar, vector):
        engine.bulk_insert_indexed(ids, indices, values)
    vector.set_adam_execution(True, 8)
    for step in range(3):
        grads = rng.normal(size=(rows, dim)).astype(np.float32)
        for engine in (scalar, vector):
            engine.adamw_tf_update_indexed(ids, grads, 0.003, 0.8, 0.95, 1e-6, 0.02)
        _assert_bits(_values(scalar, ids, dim), _values(vector, ids, dim))
    assert _state(scalar, tmp_path / "s.checkpoint")[0] == _state(
        vector, tmp_path / "v.checkpoint"
    )[0]


def test_bulk_insert_prepares_zeroed_adam_slots(tmp_path):
    lazy = _engine(tmp_path / "lazy", 1, capacity=65537, threads=4)
    engine = _engine(tmp_path / "engine", 1, capacity=65537, threads=4)
    ids = np.array([11, 22], dtype=np.int64)
    indices = np.array([0, 65536], dtype=np.int64)
    values = np.array([[0.25], [0.5]], dtype=np.float32)

    lazy_profile = lazy.bulk_insert_indexed(ids, indices, values)
    assert lazy_profile["slot_prepare_ns"] == 0
    assert lazy.optimizer_state_stats()["adam_slot_blocks"] == 0
    assert engine.optimizer_state_stats()["adam_slot_blocks"] == 0
    engine.enable_adam_slots()
    profile = engine.bulk_insert_indexed(ids, indices, values)
    stats = engine.optimizer_state_stats()
    assert profile["slot_prepare_ns"] > 0
    assert stats["adam_slots_eager"] is True
    assert stats["adam_slot_blocks"] == 2
    assert stats["adam_step"] == 0

    _, state = _state(engine, tmp_path / "prepared.checkpoint")
    _assert_bits(state["m"], np.zeros_like(state["m"]))
    _assert_bits(state["v"], np.zeros_like(state["v"]))


@pytest.mark.parametrize("ordered", [False, True])
def test_two_batch_merge_matches_explicit_coalesce(tmp_path, ordered):
    rng = np.random.default_rng(194)
    dim = 16
    ids = np.arange(1, 8, dtype=np.int64)
    indices = np.arange(ids.size, dtype=np.int64)
    values = rng.normal(size=(ids.size, dim)).astype(np.float32)
    ids_a = np.array([1, 3, 5, 7], dtype=np.int64)
    ids_b = np.array([2, 3, 6, 7], dtype=np.int64)
    if not ordered:
        ids_a = ids_a[[2, 0, 3, 1]]
        ids_b = ids_b[[3, 1, 0, 2]]
    grads_a = rng.normal(size=(ids_a.size, dim)).astype(np.float32)
    grads_b = rng.normal(size=(ids_b.size, dim)).astype(np.float32)

    reference = _engine(tmp_path / "reference", dim)
    merged = _engine(tmp_path / "merged", dim)
    for engine in (reference, merged):
        engine.bulk_insert_indexed(ids, indices, values)

    concatenated_ids = np.concatenate((ids_a, ids_b))
    concatenated_grads = np.concatenate((grads_a, grads_b))
    unique_ids, inverse = np.unique(concatenated_ids, return_inverse=True)
    reduced_grads = np.zeros((unique_ids.size, dim), dtype=np.float32)
    np.add.at(reduced_grads, inverse, concatenated_grads)
    params = (0.003, 0.8, 0.95, 1e-6, 0.02)
    for _ in range(2):
        reference.adamw_tf_update_indexed(
            unique_ids, reduced_grads, *params
        )
        profile = merged.adamw_tf_update_indexed_two_batches(
            ids_a, grads_a, ids_b, grads_b, *params
        )

    assert profile["ordered_merge"] is ordered
    assert profile["input_rows"] == 8
    assert profile["rows"] == 6
    assert profile["duplicate_rows"] == 2
    _assert_bits(_values(reference, ids, dim), _values(merged, ids, dim))
    assert _state(reference, tmp_path / "reference.checkpoint")[0] == _state(
        merged, tmp_path / "merged.checkpoint"
    )[0]


@pytest.mark.parametrize("vectorized", [False, True])
def test_missing_key_does_not_partially_update(tmp_path, vectorized):
    engine = _engine(tmp_path / "engine", 5)
    ids = np.array([11, 22], dtype=np.int64)
    values = np.arange(10, dtype=np.float32).reshape(2, 5)
    engine.bulk_insert_indexed(ids, np.array([3, 7], dtype=np.int64), values)
    engine.set_adam_execution(vectorized, 4)
    engine.adamw_tf_update_indexed(ids, np.ones_like(values), 0.003, 0.8, 0.95, 1e-6, 0.02)
    before_values = _values(engine, ids, 5)
    before_state = _state(engine, tmp_path / "before.checkpoint")[0]
    with pytest.raises(RuntimeError, match="keys must exist"):
        engine.adamw_tf_update_indexed(
            np.array([11, 999, 22], dtype=np.int64),
            np.full((3, 5), 0.7, dtype=np.float32),
            0.003, 0.8, 0.95, 1e-6, 0.02,
        )
    _assert_bits(_values(engine, ids, 5), before_values)
    assert _state(engine, tmp_path / "after.checkpoint")[0] == before_state


@pytest.mark.parametrize("distance", [-1, 65])
def test_invalid_execution_settings(tmp_path, distance):
    engine = _engine(tmp_path / "engine", 4)
    with pytest.raises(ValueError, match="prefetch_distance"):
        engine.set_adam_execution(True, distance)
