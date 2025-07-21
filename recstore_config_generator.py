import json

config = dict()

config['cache_ps'] = dict()
config['cache_ps']['ps_type'] = "GRPC"
config['cache_ps']['max_batch_keys_size'] = 65536
config['cache_ps']['num_threads'] = 32
config['cache_ps']['num_shards  '] = 2  #等于servers的数量
config['cache_ps']['servers'] = [ #host 和port根据实际情况修改
    {"host": "127.0.0.1", "port": 15000, "shard": 0},
    {"host": "127.0.0.1", "port": 15001, "shard": 1}
]

config['distributed_client'] = dict()
config['distributed_client']['num_shards'] = 2
config['distributed_client']['hash_method'] = "city_hash" ## city_hash, simple_hash
config['distributed_client']['max_keys_per_request'] = 500
config['distributed_client']['servers'] = [ #和上述servers保持一致
    {"host": "127.0.0.1", "port": 15000, "shard": 0},
    {"host": "127.0.0.1", "port": 15001, "shard": 1}
]

config['client'] = dict()
config['client']['host'] = "127.0.0.1"
config['client']['port'] = 15000
config['client']['shard'] = 1


base_kv_config = dict()
base_kv_config['kv_type'] = "KVEngineMap"
base_kv_config['path'] = "/dev/shm/" + base_kv_config['kv_type'] 
base_kv_config['capacity'] = 1 * (10**6)

# base_kv_config['value_size'] = 32
# base_kv_config['value_pool_size'] = base_kv_config['capacity'] * \
#     base_kv_config['value_size']

base_kv_config['value_pool_size'] = 1 * 1024 * 1024 * 1024

base_kv_config['corotine_per_thread'] = 1


config['cache_ps']['base_kv_config'] = base_kv_config

with open("./recstore_config.json", "w", encoding="utf-8") as f:
    json.dump(config, f, ensure_ascii=False, indent=4)