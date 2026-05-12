import json

config = {}

config["cache_ps"] = {}
config["cache_ps"]["ps_type"] = "GRPC"
config["cache_ps"]["max_batch_keys_size"] = 65536
config["cache_ps"]["num_threads"] = 32
config["cache_ps"]["num_shards"] = 2
config["cache_ps"]["servers"] = [
    {"host": "127.0.0.1", "port": 15000, "shard": 0},
    {"host": "127.0.0.1", "port": 15001, "shard": 1},
]

config["distributed_client"] = {}
config["distributed_client"]["num_shards"] = 2
config["distributed_client"]["hash_method"] = "city_hash"
config["distributed_client"]["max_keys_per_request"] = 500
config["distributed_client"]["servers"] = [
    {"host": "127.0.0.1", "port": 15000, "shard": 0},
    {"host": "127.0.0.1", "port": 15001, "shard": 1},
]

config["client"] = {}
config["client"]["host"] = "127.0.0.1"
config["client"]["port"] = 15000
config["client"]["shard"] = 1

capacity = 1_000_000
value_size = 512
base_path = "/dev/shm/recstore_kv"

config["cache_ps"]["base_kv_config"] = {
    "capacity": capacity,
    "index": {"type": "DRAM_EXTENDIBLE_HASH"},
    "value": {
        "type": "DRAM_VALUE_STORE",
        "path": f"{base_path}/value",
        "default_value_size_hint": value_size,
        "dram_allocator": {
            "type": "PERSIST_LOOP_SLAB",
            "capacity_bytes": capacity * value_size,
        },
    },
}

with open("./recstore_config.json", "w", encoding="utf-8") as f:
    json.dump(config, f, ensure_ascii=False, indent=4)
