# RecStore 配置文档

## 配置文件结构

RecStore 默认从仓库根目录的 `recstore_config.json` 读取配置。常用顶层字段有 `cache_ps`、`distributed_client`、`client`、`report_API`；HugeCTR 路径还会读取 `hugectr`，当 `hugectr.backend = "hierkv"` 时再读取 `hierkv`。

配置文件可以带 `$schema`，当前默认文件指向 `./ci/schema/recstore_config.schema.json`，方便编辑器做基础校验。运行时仍以 C++/Python 代码里的检查为准。

## 1. cache_ps 配置

`cache_ps` 是参数服务器端的配置，包含 RPC 类型、分片监听地址，以及每个分片背后的 KV 存储配置。

### 1.1 服务器基础配置

| 字段 | 类型 | 必填 | 说明 |
|------|------|------|------|
| `ps_type` | string | 是 | 参数服务器类型，可选 `"GRPC"` 或 `"BRPC"` |
| `max_batch_keys_size` | integer | 是 | 单次批量请求的最大键数量 |
| `num_threads` | integer | 是 | 服务器工作线程数 |
| `num_shards` | integer | 是 | 分片数量，应等于 `servers` 数组长度 |
| `servers` | array | 是 | 服务器节点列表 |

相关代码：

- [src/framework/op.cc](../src/framework/op.cc)、[src/ps/grpc/grpc_ps_server.cpp](../src/ps/grpc/grpc_ps_server.cpp) 和 [src/ps/brpc/brpc_ps_server.cpp](../src/ps/brpc/brpc_ps_server.cpp) 会按 `ps_type` 走不同 RPC 路径
- [src/ps/ps_server.cpp](../src/ps/ps_server.cpp) 是统一的 `ps_server` 入口
- [src/ps/grpc/grpc_ps_client.h](../src/ps/grpc/grpc_ps_client.h) 和 [src/storage/kv_engine/base_kv.h](../src/storage/kv_engine/base_kv.h) 使用批量大小等限制
- 多分片时，`servers` 里的 `shard` 字段参与路由，不要把它当成数组下标

### 1.2 servers 数组配置

每个服务器节点包含以下字段：

| 字段 | 类型 | 必填 | 说明 |
|------|------|------|------|
| `host` | string | 是 | 服务器主机地址 |
| `port` | integer | 是 | 服务器监听端口 |
| `shard` | integer | 是 | 分片编号，从 0 开始 |

???+ tip "多机环境"
    默认 `host` 是 `127.0.0.1`，只能本机连接。多机运行时，把它改成 `0.0.0.0` 或本机对外的局域网 IP；客户端侧也要使用能从训练节点访问到的地址。

### 1.3 base_kv_config 配置

`base_kv_config` 交给 [src/storage/kv_engine/engine_selector.h](../src/storage/kv_engine/engine_selector.h) 里的 `ResolveEngine` 解析。新配置优先使用嵌套的 `index` / `value` 写法；需要接入 FasterKV 或 HPS 时，再显式写 `engine_type`。

#### 推荐本地 KV 配置字段

| 字段 | 类型 | 必填 | 说明 |
|------|------|------|------|
| `path` | string | 是 | 存储路径，建议每个实例使用独立目录 |
| `capacity` | integer | 是 | 预估存储条目数 |
| `index.type` | string | 是 | `DRAM_EXTENDIBLE_HASH`、`DRAM_UNORDERED_MAP`、`DRAM_PET_HASH`、`SSD` 或 `SSD_EXTENDIBLE_HASH` |
| `value.type` | string | 是 | `DRAM_VALUE_STORE`、`SSD_VALUE_STORE` 或 `TIERED_VALUE_STORE` |
| `value.default_value_size_hint` | integer | 建议 | 单条值的字节数 |

本地 value store 的 allocator 要和类型匹配：

- `DRAM_VALUE_STORE` 需要 `value.dram_allocator`
- `SSD_VALUE_STORE` 需要 `value.ssd_allocator`
- `TIERED_VALUE_STORE` 同时需要 `value.dram_allocator` 和 `value.ssd_allocator`

#### 兼容旧版本地 KV 配置字段

旧配置里的 `index_type` / `value_type` 还在兼容，主要服务已有 DRAM+SSD 配置、本地 CI 和历史实验。`ResolveEngine` 会先把它们归一到当前的 `index` / `value` 结构，再创建本地 `KVEngine`。如果配置里显式写了外部 `engine_type`，这条兼容路径不会抢走外部 engine。

| 字段 | 类型 | 必填 | 说明 |
|------|------|------|------|
| `path` | string | 是 | 存储路径 |
| `index_type` | string | 是 | `DRAM` 或 `SSD` |
| `value_type` | string | 是 | `DRAM`、`SSD` 或 `HYBRID` |
| `capacity` | integer | DRAM/SSD 需要 | 预估存储条目数 |
| `value_size` | integer | DRAM/SSD 需要 | 固定 value 字节数 |
| `shmcapacity` | integer | HYBRID 需要 | DRAM 层字节数 |
| `ssdcapacity` | integer | HYBRID 需要 | SSD 层字节数 |
| `value_memory_management` | string | 否 | `PersistLoopShmMalloc` 或 `R2ShmMalloc` |

#### 外部 KV engine 配置字段

| 字段 | 类型 | 必填 | 说明 |
|------|------|------|------|
| `engine_type` | string | 是 | `KVEngineFasterKV`、`KVEngineHPSHashMap` 或 `KVEngineHPSRocksDB` |
| `path` | string | 是 | 存储路径或工作目录 |
| `capacity` | integer | 是 | 预估存储条目数 |
| `value_size` | integer | 是 | 固定 value 字节数；也可用 `value.default_value_size_hint` |
| `max_batch_size` | integer | 否 | HPS/FasterKV 批处理上限 |
| `table_name` | string | 否 | HPS table name，默认 `default` |
| `rocksdb_path` | string | 否 | HPS RocksDB 数据目录，默认 `${path}/hps_rocksdb` |

#### 可选字段

| 字段 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `value_memory_management` | string | `"PersistLoopShmMalloc"` | 内存管理器类型：`"PersistLoopShmMalloc"` 或 `"R2ShmMalloc"` |

内存管理器选择在 [src/storage/hybrid/value.h](../src/storage/hybrid/value.h) 生效。做本地 benchmark bring-up 时要留意：当前 `rs_demo` 路径下，`R2ShmMalloc` 可能触发 `ps_server` 不稳定；只想先拿到可运行 baseline 时，可以临时使用 `PersistLoopShmMalloc`，但报告里要写清楚。

#### 引擎类型自动推导规则

`base::ResolveEngine` 默认走本地 `KVEngine`，再根据 `index.type` 和 `value.type` 选择具体 index/value store。写了 `engine_type` 时，会按对应的 `BaseKV` factory key 创建外部 engine。旧的 `index_type` / `value_type` 不是废弃字段，仍用于兼容历史 DRAM+SSD 配置；新文件建议写嵌套结构。

引擎推导实现位置：[src/storage/kv_engine/engine_selector.h](../src/storage/kv_engine/engine_selector.h)

## 2. distributed_client 配置

`distributed_client` 是分布式客户端的路由配置。多分片读写时，客户端按这里的 `hash_method` 把 key 分到 shard，再按 `servers` 里的显式 `shard` id 找到目标节点。

| 字段 | 类型 | 必填 | 说明 |
|------|------|------|------|
| `num_shards` | integer | 是 | 分片总数，应与 `cache_ps.num_shards` 一致 |
| `hash_method` | string | 是 | 哈希方法：`"city_hash"` 或 `"simple_mod"` |
| `max_keys_per_request` | integer | 否（默认 500） | 单次请求最大键数量 |
| `servers` | array | 是 | 服务器节点配置，结构同 `cache_ps.servers` |

相关代码在 [src/ps/brpc/dist_brpc_ps_client.cpp](../src/ps/brpc/dist_brpc_ps_client.cpp) 和 [src/ps/grpc/dist_grpc_ps_client.cpp](../src/ps/grpc/dist_grpc_ps_client.cpp)。`GetShardId`、`PartitionKeys` 决定 key 到 shard 的映射，`InitializeClients` 根据 `servers` 建立每个 shard 的客户端。

注意两点：

- 分布式客户端以 `distributed_client.servers` 为准，不要默认复用 `cache_ps.servers`
- `shard` 是配置里的显式 id，不等于 `servers` 数组排序后的下标

## 3. client 配置

`client` 是单节点客户端配置，用来直接连一个参数服务器实例。

| 字段 | 类型 | 必填 | 说明 |
|------|------|------|------|
| `host` | string | 是 | 服务器主机地址 |
| `port` | integer | 是 | 服务器端口 |
| `shard` | integer | 是 | 目标分片编号 |

单节点客户端会在 [src/ps/grpc/grpc_ps_client.cpp](../src/ps/grpc/grpc_ps_client.cpp) 读取这些字段，创建 gRPC 通道，并把分片元数据保存在 [src/ps/grpc/grpc_ps_client.h](../src/ps/grpc/grpc_ps_client.h)。

???+ tip "动态覆盖"
    多机训练时，Python 侧可以通过 `client.set_ps_config(host, port)` 覆盖配置文件里的 `host` 和 `port`。这样不用为每个训练节点改一份 `recstore_config.json`。接口说明见 [KVClient Python 客户端](./calc/kvclient.md)。

## 4. report_API 配置

`report_API` 配置性能事件的 HTTP 上报地址。

| 字段 | 类型 | 必填 | 默认值 | 说明 |
|------|------|------|--------|------|
| `report_API` | string | 否 | `http://127.0.0.1:8081/report` | `report()` 异步上报接口地址，通常由该服务写入 ClickHouse 并供 Grafana 展示 |

[src/base/report/report_client.cpp](../src/base/report/report_client.cpp) 读取这个字段。缺失时默认使用 `http://127.0.0.1:8081/report`。

## 5. hugectr 配置

`hugectr` 只影响 HugeCTR 入口的运行时 backend 选择，不改变 `CommonOp` 的语义。分发逻辑在 [src/framework/hugectr/op_hugectr.cc](../src/framework/hugectr/op_hugectr.cc)。

| 字段 | 类型 | 必填 | 默认值 | 说明 |
|------|------|------|--------|------|
| `backend` | string | 否 | `recstore` | HugeCTR 运行时后端，可选 `recstore` 或 `hierkv` |

当前两个取值：

- `recstore`：使用现有 host staging + RecStore OP 路径
- `hierkv`：切到 HugeCTR 专用 HierKV backend 适配层

## 6. hierkv 配置

`hierkv` 只在 `hugectr.backend = "hierkv"` 时读取，用来初始化 HugeCTR 专用的 HierarchicalKV backend。

| 字段 | 类型 | 必填 | 说明 |
|------|------|------|------|
| `max_capacity` | integer | 是 | HierarchicalKV 最大容量 |
| `max_hbm_for_vectors` | integer | 是 | 向量可使用的 HBM 上限，单位字节 |
| `dim` | integer | 是 | embedding 维度 |
| `device_id` | integer | 否 | 可选 CUDA 设备号 |
| `reserved_key_start_bit` | integer | 否 | 可选保留 key 起始 bit，范围 0-62 |

注意：

- 这组字段只进入 HugeCTR 侧适配层，不回流到 `CommonOp`
- 选择 `hierkv` 但缺少必填字段时，HugeCTR 入口会直接报配置错误

## 配置示例

???+ example "完整配置示例"

    ```json
    {
        "cache_ps": {
            "ps_type": "GRPC",
            "max_batch_keys_size": 65536,
            "num_threads": 32,
            "num_shards": 2,
            "servers": [
                {
                    "host": "127.0.0.1",
                    "port": 15000,
                    "shard": 0
                },
                {
                    "host": "127.0.0.1",
                    "port": 15001,
                    "shard": 1
                }
            ],
            "base_kv_config": {
                "path": "/tmp/recstore_data",
                "capacity": 40000000,
                "index": {
                    "type": "DRAM_EXTENDIBLE_HASH"
                },
                "value": {
                    "type": "DRAM_VALUE_STORE",
                    "default_value_size_hint": 512,
                    "dram_allocator": {
                        "type": "PERSIST_LOOP_SLAB",
                        "capacity_bytes": 20480000000
                    }
                }
            }
        },
        "distributed_client": {
            "num_shards": 2,
            "hash_method": "city_hash",
            "max_keys_per_request": 500,
            "servers": [
                {
                    "host": "127.0.0.1",
                    "port": 15000,
                    "shard": 0
                },
                {
                    "host": "127.0.0.1",
                    "port": 15001,
                    "shard": 1
                }
            ]
        },
        "client": {
            "host": "127.0.0.1",
            "port": 15000,
            "shard": 0
        },
        "report_API": "http://127.0.0.1:8081/report"
    }
    ```

??? example "FasterKV 外部 KV 配置"
    ```json
    {
        "cache_ps": {
            "base_kv_config": {
                "engine_type": "KVEngineFasterKV",
                "path": "/data/fasterkv",
                "capacity": 2000000,
                "value_size": 128,
                "max_batch_size": 65536
            }
        }
    }
    ```

??? example "HPS HashMap 外部 KV 配置"
    ```json
    {
        "cache_ps": {
            "base_kv_config": {
                "engine_type": "KVEngineHPSHashMap",
                "path": "/data/hps_hash_map",
                "capacity": 2000000,
                "value_size": 128,
                "max_batch_size": 65536,
                "table_name": "default"
            }
        }
    }
    ```

??? example "HPS RocksDB 外部 KV 配置"
    ```json
    {
        "cache_ps": {
            "base_kv_config": {
                "engine_type": "KVEngineHPSRocksDB",
                "path": "/data/hps",
                "rocksdb_path": "/data/hps/rocksdb",
                "capacity": 2000000,
                "value_size": 128,
                "max_batch_size": 65536,
                "table_name": "default"
            }
        }
    }
    ```

??? example "本地 Tiered Value Store 配置"

    ```json
    {
        "cache_ps": {
            "base_kv_config": {
                "path": "/data/recstore",
                "capacity": 2000000,
                "index": {
                    "type": "DRAM_EXTENDIBLE_HASH"
                },
                "value": {
                    "type": "TIERED_VALUE_STORE",
                    "default_value_size_hint": 128,
                    "dram_allocator": {
                        "type": "PERSIST_LOOP_SLAB",
                        "capacity_bytes": 10737418240
                    },
                    "ssd_allocator": {
                        "type": "SSD_BUDDY",
                        "capacity_bytes": 107374182400,
                        "io": {
                            "type": "IOURING",
                            "file_path": "/data/recstore/value.db",
                            "queue_depth": 512
                        }
                    }
                }
            }
        }
    }
    ```

??? example "CI 中的轻量配置"

    GitHub Actions 的 CPU 资源有限，也没有 GPU。CI 脚本会把配置压到更小规模，例如：

    ```shell
    jq '.cache_ps.base_kv_config.capacity = 512
        | .cache_ps.max_batch_keys_size = 128
        | .cache_ps.num_threads = 4
        | .distributed_client.max_keys_per_request = 32
        | .cache_ps.base_kv_config.index_type = "DRAM"
        | .cache_ps.base_kv_config.value_type = "SSD"
        | .cache_ps.base_kv_config.type = "DRAM"
        | .cache_ps.base_kv_config.queue_size = 1024'
    ```

    这组覆盖会使用 DRAM 索引 + SSD value，并降低容量和批量大小，避免 CI 资源被默认配置吃满。

## 配置文件使用

配置文件通常保存为 `recstore_config.json`。常见读取方式如下：

```cpp
std::ifstream config_file("recstore_config.json");
nlohmann::json config;
config_file >> config;

// 服务器端使用
auto cache_ps = std::make_unique<CachePS>(config["cache_ps"]);

// 分布式客户端使用
auto dist_client = std::make_unique<DistributedBRPCParameterClient>(config);

// 单节点客户端使用
auto client = std::make_unique<GRPCParameterClient>(config["client"]);
```
