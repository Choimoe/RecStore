"""Stable-row allocation contracts for ordered misses and hash fallback."""

import numpy as np
import pytest

import _recstore_kv


ENTRIES = ("numpy", "raw", "raw_legacy")
DIM = 3


def _engine(path, capacity=64, threads=1):
    return _recstore_kv.LocalIndexedKV(
        capacity=capacity,
        embedding_dim=DIM,
        index_type="DRAM_PET_HASH",
        value_path=str(path),
        allocator_capacity=4 << 20,
        num_threads=threads,
    )


def _lookup(engine, entry, ids, allocate=True):
    ids = np.ascontiguousarray(ids, dtype=np.int64)
    if entry == "numpy":
        return engine.lookup_indexed_profiled(ids, allocate)
    indices = np.full(ids.size, -99, dtype=np.int64)
    values = np.full((ids.size, DIM), np.nan, dtype=np.float32)
    found = np.zeros(ids.size, dtype=np.bool_)
    method = (engine.lookup_indexed_into_raw if entry == "raw"
              else engine.lookup_indexed_legacy_into_raw)
    profile = method(ids.ctypes.data, ids.size, indices.ctypes.data,
                     values.ctypes.data, found.ctypes.data, allocate, True, True)
    return indices, values, found, profile


def _check_profile(profile, fast_path, proof=None):
    assert profile["missing_allocate_fast_path"] is fast_path
    if proof is None:
        proof = "signed_int64" if fast_path else "none"
    assert profile["missing_allocate_proof"] == proof
    # Presence/type only: correctness must not depend on a timer threshold.
    assert isinstance(profile["missing_allocate_ns"], (int, float))
    assert profile["missing_allocate_ns"] >= 0


def _values(indices):
    return (np.asarray(indices, dtype=np.float32)[:, None] * np.float32(0.5)
            + np.arange(DIM, dtype=np.float32)[None, :])


def _partition_order(ids):
    # ids_partition_kernel.cu applies HashKey<int64_t>{16}, then sorts through
    # a uint64_t pointer. NumPy's uint64 sort models that exact order on CPU.
    ids = np.asarray(ids, dtype=np.int64)
    keys = ids.view(np.uint64)
    hashes = (keys << np.uint64(48)) | (keys >> np.uint64(16))
    order = np.argsort(hashes, kind="stable")
    return np.ascontiguousarray(ids[order]), hashes[order]


def _commit(engine, ids, indices):
    ids = np.asarray(ids, dtype=np.int64)
    indices = np.asarray(indices, dtype=np.int64)
    unique, first = np.unique(indices, return_index=True)
    engine.bulk_insert_indexed(np.ascontiguousarray(ids[first]), unique, _values(unique))


def _snapshot(engine):
    ids, indices, values = engine.snapshot_indexed()
    order = np.argsort(ids)
    return ids[order], indices[order], values[order]


def _checkpoint_state(engine, path):
    assert engine.save_checkpoint(str(path), "missing-allocation-contract")
    files = list(path.parent.glob(path.name + ".*.state"))
    assert len(files) == 1
    return files[0].read_bytes()


@pytest.mark.parametrize("entry", ENTRIES)
@pytest.mark.parametrize("ids, expected, fast_path", [
    ([-11, -3, 0, 4, 100], [0, 1, 2, 3, 4], True),
    ([-(1 << 63), -1, 0, (1 << 63) - 1], [0, 1, 2, 3], True),
    # Increasing uint64 values are not necessarily increasing signed int64.
    ([0, 9, -(1 << 63), -1], [0, 1, 2, 3], False),
    ([-3, -3, 1, 1], [0, 0, 1, 1], False),
    ([4, -3, 4, 0], [0, 1, 0, 2], False),
    ([7], [0], True),
])
def test_sorted_unique_and_fallback_preserve_first_occurrence(
    tmp_path, entry, ids, expected, fast_path,
):
    engine = _engine(tmp_path / "engine")
    indices, values, found, profile = _lookup(engine, entry, ids)
    np.testing.assert_array_equal(indices, expected)
    np.testing.assert_array_equal(found, np.zeros(len(ids), dtype=np.bool_))
    np.testing.assert_array_equal(values, np.zeros((len(ids), DIM), dtype=np.float32))
    _check_profile(profile, fast_path)
    assert engine.size() == 0  # Allocation does not publish a key/value.
    assert engine.storage_stats()["stable_rows"] == len(set(ids))
    _commit(engine, ids, indices)
    actual_indices, actual_values, found, profile = _lookup(engine, entry, ids)
    np.testing.assert_array_equal(actual_indices, expected)
    np.testing.assert_array_equal(actual_values, _values(expected))
    assert found.all()
    _check_profile(profile, False)


@pytest.mark.parametrize("entry", ENTRIES)
def test_only_missing_subsequence_must_be_ordered(tmp_path, entry):
    engine = _engine(tmp_path / "engine")
    seed_ids = np.array([7, -99], dtype=np.int64)
    seed_indices = np.array([0, 1], dtype=np.int64)
    engine.bulk_insert_indexed(seed_ids, seed_indices, _values(seed_indices))
    ids = [7, -(1 << 63), 7, -3, -99, 0, (1 << 63) - 1]
    indices, values, found, profile = _lookup(engine, entry, ids)
    expected = [0, 2, 0, 3, 1, 4, 5]
    np.testing.assert_array_equal(indices, expected)
    np.testing.assert_array_equal(found, [True, False, True, False, True, False, False])
    np.testing.assert_array_equal(values[found], _values(np.asarray(expected)[found]))
    np.testing.assert_array_equal(values[~found], np.zeros((4, DIM), dtype=np.float32))
    _check_profile(profile, True)
    assert engine.storage_stats()["stable_rows"] == 6


@pytest.mark.parametrize("ids, proof", [
    ([-8, 0, 2], "signed_int64"),
    ([65536, -65536, 1], "quanta_partition_uint64"),
])
def test_entry_points_preserve_lifo_reuse_and_checkpoint_state(tmp_path, ids, proof):
    results = []
    states = []
    for entry in ENTRIES:
        engine = _engine(tmp_path / entry)
        seed_ids = np.array([-2, 5, 9, 10], dtype=np.int64)
        seed_indices = np.arange(4, dtype=np.int64)
        engine.bulk_insert_indexed(seed_ids, seed_indices, _values(seed_indices))
        assert engine.delete_indexed(np.array([5, 10], dtype=np.int64)) == 2
        assert engine.storage_stats()["free_rows"] == 2
        indices, values, found, profile = _lookup(engine, entry, ids)
        # Deletion pushed [1, 3]; the ordered fast path must still pop 3, then 1.
        np.testing.assert_array_equal(indices, [3, 1, 4])
        _check_profile(profile, True, proof)
        assert not found.any()
        np.testing.assert_array_equal(values, np.zeros((3, DIM), dtype=np.float32))
        assert engine.storage_stats()["stable_rows"] == 5
        assert engine.storage_stats()["free_rows"] == 0
        _commit(engine, ids, indices)
        results.append(_snapshot(engine))
        states.append(_checkpoint_state(engine, tmp_path / (entry + ".checkpoint")))
    for result in results[1:]:
        for expected, actual in zip(results[0], result):
            np.testing.assert_array_equal(actual, expected)
    assert states[0] == states[1] == states[2]


@pytest.mark.parametrize("entry", ENTRIES)
@pytest.mark.parametrize("ids", [
    [-3, -1, 8, 9], [8, -1, -3, 9], [65536, -65536, 1, 2],
])
def test_capacity_failure_keeps_existing_partial_reservation_semantics(tmp_path, entry, ids):
    engine = _engine(tmp_path / "engine", capacity=4)
    engine.bulk_insert_indexed(np.array([100], dtype=np.int64),
                               np.array([0], dtype=np.int64), _values([0]))
    with pytest.raises(RuntimeError, match="row-index capacity exceeded"):
        _lookup(engine, entry, ids)
    stats = engine.storage_stats()
    assert stats["stable_rows"] == 4
    assert stats["free_rows"] == 0
    assert stats["active_rows"] == 1
    # No missing key was inserted, even though three row indices were consumed.
    indices, values, found, profile = _lookup(engine, entry, ids, allocate=False)
    np.testing.assert_array_equal(indices, [-1] * 4)
    np.testing.assert_array_equal(values, np.zeros((4, DIM), dtype=np.float32))
    assert not found.any()
    _check_profile(profile, False)
    assert engine.delete_indexed(np.array([100], dtype=np.int64)) == 1
    with pytest.raises(RuntimeError, match="row-index capacity exceeded"):
        _lookup(engine, entry, [-9, -8])
    # The freed row was popped before capacity failed again, as in the old loop.
    assert engine.storage_stats()["free_rows"] == 0
    assert engine.storage_stats()["stable_rows"] == 4


@pytest.mark.parametrize("entry", ENTRIES)
def test_empty_and_readonly_do_not_reserve_rows(tmp_path, entry):
    engine = _engine(tmp_path / "engine")
    for ids, allocate in (([], True), ([-9, 0, 9], False)):
        indices, values, found, profile = _lookup(engine, entry, ids, allocate)
        np.testing.assert_array_equal(indices, [-1] * len(ids))
        np.testing.assert_array_equal(values, np.zeros((len(ids), DIM), dtype=np.float32))
        assert not found.any()
        _check_profile(profile, False)
        assert engine.storage_stats()["stable_rows"] == 0
        assert engine.size() == 0


@pytest.mark.parametrize("proof", ["signed_int64", "quanta_partition_uint64"])
def test_parallel_lookup_entry_points_allocate_in_input_order(tmp_path, proof):
    ids = np.arange(-4096, 4097, dtype=np.int64)
    if proof == "quanta_partition_uint64":
        ids, _ = _partition_order(ids)
    expected = np.arange(ids.size, dtype=np.int64)
    for entry in ENTRIES:
        engine = _engine(tmp_path / entry, capacity=16384, threads=4)
        indices, values, found, profile = _lookup(engine, entry, ids)
        np.testing.assert_array_equal(indices, expected)
        np.testing.assert_array_equal(values, np.zeros((ids.size, DIM), dtype=np.float32))
        assert not found.any()
        _check_profile(profile, True, proof)
        assert engine.storage_stats()["stable_rows"] == ids.size


@pytest.mark.parametrize("entry", ENTRIES)
@pytest.mark.parametrize("duplicate", [False, True])
def test_partition_order_crosses_signed_boundaries_without_reordering(
    tmp_path, entry, duplicate,
):
    ids, hashes = _partition_order([
        -(1 << 63), -(1 << 63) + 1, -65536, -65535, -32768, -1,
        0, 1, 32767, 32768, 65535, 65536, 65537, (1 << 63) - 1,
    ])
    assert np.all(hashes[1:] > hashes[:-1])
    assert hashes[0] < np.uint64(1 << 63) <= hashes[-1]
    assert np.any(ids[1:] < ids[:-1])  # Signed-order proof must fail.
    if duplicate:
        middle = len(ids) // 2
        ids = np.insert(ids, middle, ids[middle])
    expected = []
    first_occurrence = {}
    for key in ids.tolist():
        if key not in first_occurrence:
            first_occurrence[key] = len(first_occurrence)
        expected.append(first_occurrence[key])
    engine = _engine(tmp_path / "engine")
    indices, values, found, profile = _lookup(engine, entry, ids)
    np.testing.assert_array_equal(indices, expected)
    np.testing.assert_array_equal(values, np.zeros((ids.size, DIM), dtype=np.float32))
    assert not found.any()
    _check_profile(profile, not duplicate,
                   "none" if duplicate else "quanta_partition_uint64")
    assert engine.storage_stats()["stable_rows"] == len(first_occurrence)
    _commit(engine, ids, indices)
    actual, values, found, profile = _lookup(engine, entry, ids, allocate=False)
    np.testing.assert_array_equal(actual, expected)
    np.testing.assert_array_equal(values, _values(expected))
    assert found.all()
    _check_profile(profile, False)


@pytest.mark.parametrize("entry", ENTRIES)
def test_partition_proof_ignores_unsorted_repeated_hits(tmp_path, entry):
    engine = _engine(tmp_path / "engine")
    seed_ids = np.array([12345678, -12345678], dtype=np.int64)
    seed_indices = np.array([0, 1], dtype=np.int64)
    engine.bulk_insert_indexed(seed_ids, seed_indices, _values(seed_indices))
    missing, _ = _partition_order([-65536, 0, 1, -1, 32768, 65536])
    ids = np.concatenate((seed_ids[:1], missing[:2], seed_ids[::-1],
                          missing[2:], seed_ids[1:]))
    known = {int(key): int(index) for key, index in zip(seed_ids, seed_indices)}
    expected = dict(known)
    expected.update({int(key): i + 2 for i, key in enumerate(missing)})
    indices, values, found, profile = _lookup(engine, entry, ids)
    np.testing.assert_array_equal(indices, [expected[int(key)] for key in ids])
    np.testing.assert_array_equal(found, [int(key) in known for key in ids])
    np.testing.assert_array_equal(values[found], _values(indices[found]))
    np.testing.assert_array_equal(values[~found], np.zeros((missing.size, DIM), dtype=np.float32))
    _check_profile(profile, True, "quanta_partition_uint64")
    assert engine.storage_stats()["stable_rows"] == missing.size + 2
