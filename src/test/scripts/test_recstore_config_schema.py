import json
import unittest
from pathlib import Path

import jsonschema


REPO_ROOT = Path(__file__).resolve().parents[3]
SCHEMA_PATH = REPO_ROOT / "ci" / "schema" / "recstore_config.schema.json"


class TestRecstoreConfigSchema(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.schema = json.loads(SCHEMA_PATH.read_text(encoding="utf-8"))

    def validate_config(self, base_kv_config: dict) -> None:
        config = {
            "cache_ps": {
                "ps_type": "GRPC",
                "max_batch_keys_size": 1024,
                "num_threads": 4,
                "num_shards": 1,
                "servers": [
                    {"host": "127.0.0.1", "port": 15000, "shard": 0},
                ],
                "base_kv_config": base_kv_config,
            },
            "distributed_client": {
                "num_shards": 1,
                "hash_method": "city_hash",
                "servers": [
                    {"host": "127.0.0.1", "port": 15000, "shard": 0},
                ],
            },
            "client": {"host": "127.0.0.1", "port": 15000, "shard": 0},
        }
        jsonschema.Draft202012Validator(self.schema).validate(config)

    def test_accepts_nested_local_kv_config(self) -> None:
        self.validate_config(
            {
                "path": "/tmp/recstore_data",
                "capacity": 1024,
                "index": {"type": "DRAM_EXTENDIBLE_HASH"},
                "value": {
                    "type": "DRAM_VALUE_STORE",
                    "default_value_size_hint": 128,
                    "dram_allocator": {
                        "type": "PERSIST_LOOP_SLAB",
                        "capacity_bytes": 1024 * 128,
                    },
                },
            }
        )

    def test_accepts_legacy_dram_ssd_kv_config(self) -> None:
        self.validate_config(
            {
                "path": "/tmp/recstore_data",
                "capacity": 1024,
                "index_type": "DRAM",
                "value_type": "SSD",
                "value_size": 128,
                "value_memory_management": "PersistLoopShmMalloc",
                "queue_size": 64,
            }
        )

    def test_accepts_external_fasterkv_config(self) -> None:
        self.validate_config(
            {
                "engine_type": "KVEngineFasterKV",
                "path": "/tmp/fasterkv_data",
                "capacity": 1024,
                "value_size": 128,
                "max_batch_size": 64,
            }
        )

    def test_accepts_external_hps_rocksdb_config(self) -> None:
        self.validate_config(
            {
                "engine_type": "KVEngineHPSRocksDB",
                "path": "/tmp/hps_data",
                "rocksdb_path": "/tmp/hps_rocksdb",
                "capacity": 1024,
                "value_size": 128,
                "table_name": "default",
            }
        )


if __name__ == "__main__":
    unittest.main()
