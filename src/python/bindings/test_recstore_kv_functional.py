import tempfile

import numpy as np

import _recstore_kv


with tempfile.TemporaryDirectory(prefix="recstore_kv_binding_") as root:
    engine = _recstore_kv.LocalIndexedKV(
        capacity=1024,
        embedding_dim=16,
        index_type="DRAM_PET_HASH",
        value_path=root,
        allocator="CONCURRENT_SLAB_MEMORY_POOL",
        allocator_capacity=64 << 20,
        num_threads=4,
    )
    seed_ids = np.array([17, 91], dtype=np.int64)
    seed_indices = np.array([3, 7], dtype=np.int64)
    seed_values = np.stack(
        [np.arange(16, dtype=np.float32), np.arange(16, dtype=np.float32) + 100]
    )
    engine.bulk_insert_indexed(seed_ids, seed_indices, seed_values)

    query = np.array([17, 42, 91, 43], dtype=np.int64)
    values = np.empty((query.size, 16), dtype=np.float32)
    missing_positions = np.full(query.size, -9, dtype=np.int64)
    missing = engine.lookup_values_with_misses_into_raw(
        query.ctypes.data,
        query.size,
        values.ctypes.data,
        missing_positions.ctypes.data,
    )
    assert missing == 2
    np.testing.assert_array_equal(missing_positions[:missing], [1, 3])
    np.testing.assert_array_equal(values[0], seed_values[0])
    np.testing.assert_array_equal(values[1], np.zeros(16, dtype=np.float32))
    np.testing.assert_array_equal(values[2], seed_values[1])
    np.testing.assert_array_equal(values[3], np.zeros(16, dtype=np.float32))

    profile = engine.lookup_values_with_misses_into_raw_profiled(
        query.ctypes.data,
        query.size,
        values.ctypes.data,
        missing_positions.ctypes.data,
    )
    assert profile["missing_rows"] == 2
    assert profile["missing_positions_compact_ns"] >= 0

    indices = np.empty(query.size, dtype=np.int64)
    found = np.empty(query.size, dtype=np.bool_)
    indexed_profile = engine.lookup_indexed_into_raw(
        query.ctypes.data,
        query.size,
        indices.ctypes.data,
        values.ctypes.data,
        found.ctypes.data,
        False,
    )
    np.testing.assert_array_equal(indices, [3, -1, 7, -1])
    np.testing.assert_array_equal(found, [True, False, True, False])
    assert indexed_profile["missing_rows"] == 2
    assert engine.size() == 2
