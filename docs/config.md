# RecStore 配置文档

## 配置文件结构

RecStore 配置采用 JSON 格式，位于根目录。当前仓库默认包含以下顶层配置块：`cache_ps`、`distributed_client`、`client`、`report_API`，以及按需启用的 `hugectr`、`hierkv`。

## 1. cache_ps 配置

`cache_ps` 配置用于参数服务器（Parameter Server）端，定义服务器的运行参数和底层存储引擎。

### 1.1 服务器基础配置

| 字段 | 类型 | 必填 | 说明 |
|------|------|------|------|
| `ps_type` | string | 是 | 参数服务器类型，可选 `"GRPC"` 或 `"BRPC"` |
| `max_batch_keys_size` | integer | 是 | 单次批量请求的最大键数量 |
| `num_threads` | integer | 是 | 服务器工作线程数 |
| `num_shards` | integer | 是 | 分片数量，应等于 `servers` 数组长度 |
| `servers` | array | 是 | 服务器节点配置数组 |

作用位置：通信协议选择见 [src/framework/op.cc](../src/framework/op.cc)、[src/grpc_ps/grpc_ps_server.cpp](../src/grpc_ps/grpc_ps_server.cpp)。同时也可通过统一入口可执行程序 [src/grpc_ps/ps_server.cpp](../src/grpc_ps/ps_server.cpp) 按 `ps_type` 自动选择 GRPC 或 bRPC；批量限制与线程数在 [src/grpc_ps/grpc_ps_client.h](../src/grpc_ps/grpc_ps_client.h) 及 [src/storage/kv_engine/base_kv.h](../src/storage/kv_engine/base_kv.h) 中生效；分片与 servers 列表用于分布式路由。

### 1.2 servers 数组配置

每个服务器节点包含以下字段：

| 字段 | 类型 | 必填 | 说明 |
|------|------|------|------|
| `host` | string | 是 | 服务器主机地址 |
| `port` | integer | 是 | 服务器监听端口 |
| `shard` | integer | 是 | 分片编号，从 0 开始

???+ tip "多机环境"
    默认配置中 `host` 为 `127.0.0.1`，仅允许本地访问。在多机场景下，必须修改为 `0.0.0.0` 或本机对外的局域网 IP (LAN IP)，以允许远程客户端连接。

### 1.3 base_kv_config 配置

`base_kv_config` 定义底层键值存储引擎的配置，由 [src/storage/kv_engine/engine_selector.h](../src/storage/kv_engine/engine_selector.h) 的 `ResolveEngine` 按 `index.type` 和 `value.type` 校验并解析。

#### 通用必填字段（非 HYBRID 模式）

| 字段 | 类型 | 必填 | 说明 |
|------|------|------|------|
| `capacity` | integer | 是 | Key 容量上限 |
| `index` | object | 是 | 包含 `type`，SSD 索引需提供 `index.path` |
| `value` | object | 是 | 包含 `type`，单层存储使用 `value.path`，TIERED 使用 allocator path |
| `capacity` | integer | 是 | 预估存储条目数 |
| `value_size` | integer | 是 | 单条值的字节数 |

作用位置：字段最终进入 `BaseKVConfig.json_config_`，在 [src/storage/kv_engine/engine_selector.h](../src/storage/kv_engine/engine_selector.h) 中推导引擎类型，并在 [src/storage/kv_engine/base_kv.h](../src/storage/kv_engine/base_kv.h) 及对应引擎实现中用于容量预分配与值大小约束。

#### HYBRID 模式必填字段

| 字段 | 类型 | 必填 | 说明 |
|------|------|------|------|
| `shmcapacity` | integer | 是 | DRAM/共享内存侧的字节数容量 |
| `ssdcapacity` | integer | 是 | SSD 侧的字节数容量 |

???+ note "注意"
    HYBRID 模式不需要 `capacity` 和 `value_size`。

#### 可选字段

| 字段 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `value_memory_management` | string | `"PersistLoopShmMalloc"` | 内存管理器类型：`"PersistLoopShmMalloc"` 或 `"R2ShmMalloc"` |

作用位置：内存管理器选择在 [src/storage/hybrid/value.h](../src/storage/hybrid/value.h) 生效，决定底层分配策略。

#### 引擎类型自动推导规则

`base::ResolveEngine` 根据 `index.type` 和 `value.type` 组合解析引擎：

- `index.type` 取值：`DRAM_EXTENDIBLE_HASH` / `DRAM_UNORDERED_MAP` / `DRAM_PET_HASH` / `SSD` / `SSD_EXTENDIBLE_HASH`
- `value.type` 取值：`DRAM_VALUE_STORE` / `SSD_VALUE_STORE` / `TIERED_VALUE_STORE`
- 旧字段 `path`、`index.io.file_path`、`value.ssd_allocator.io.file_path` 会被拒绝

引擎推导实现位置：[src/storage/kv_engine/engine_selector.h](../src/storage/kv_engine/engine_selector.h)

## 2. distributed_client 配置

`distributed_client` 配置用于分布式客户端，实现多分片参数服务器的访问。

| 字段 | 类型 | 必填 | 说明 |
|------|------|------|------|
| `num_shards` | integer | 是 | 分片总数，应与 `cache_ps.num_shards` 一致 |
| `hash_method` | string | 是 | 哈希方法：`"city_hash"` 或 `"simple_mod"` |
| `max_keys_per_request` | integer | 否（默认 500） | 单次请求最大键数量 |
| `servers` | array | 是 | 服务器节点配置，结构同 `cache_ps.servers` |

作用位置：分片数与哈希方法在 [src/grpc_ps/dist_brpc_ps_client.cpp](../src/grpc_ps/dist_brpc_ps_client.cpp) 与 [src/grpc_ps/dist_grpc_ps_client.cpp](../src/grpc_ps/dist_grpc_ps_client.cpp) 的 `GetShardId`、`PartitionKeys` 中决定路由；`max_keys_per_request` 限制单分片请求大小；`servers` 列表在 `InitializeClients` 中生成各分片客户端。

## 3. client 配置

`client` 配置用于单节点客户端，直接连接单个参数服务器。

| 字段 | 类型 | 必填 | 说明 |
|------|------|------|------|
| `host` | string | 是 | 服务器主机地址 |
| `port` | integer | 是 | 服务器端口 |
| `shard` | integer | 是 | 目标分片编号 |

作用位置：单节点客户端在 [src/grpc_ps/grpc_ps_client.cpp](../src/grpc_ps/grpc_ps_client.cpp) 读取以上字段创建 gRPC 通道并标识分片，在 [src/grpc_ps/grpc_ps_client.h](../src/grpc_ps/grpc_ps_client.h) 中保存元数据。

???+ tip "动态覆盖"
    在实际运行时（尤其是在多机训练场景下），可以通过 KVClient 动态覆盖配置文件的 `host` 与 `port` 设置 `client.set_ps_config(host, port)`，这允许在不修改 `recstore_config.json` 的情况下灵活调整连接目标。你可以在 [KVClient Python 客户端](./calc/kvclient.md) 中找到详细信息。

## 4. report_API 配置

`report_API` 是顶层配置项，用于指定性能 report 的 HTTP 上报入口。

| 字段 | 类型 | 必填 | 默认值 | 说明 |
|------|------|------|--------|------|
| `report_API` | string | 否 | `http://127.0.0.1:8081/report` | `report()` 异步上报接口地址，通常由该服务写入 ClickHouse 并供 Grafana 展示 |

作用位置：`report_API` 由 [src/base/report/report_client.cpp](../src/base/report/report_client.cpp) 读取；当该字段缺失时会回退到默认地址 `http://127.0.0.1:8081/report`。

## 5. hugectr 配置

`hugectr` 顶层配置用于 HugeCTR 入口的运行时后端选择。它不会修改 `CommonOp` 语义，而是只影响 [src/framework/hugectr/op_hugectr.cc](../src/framework/hugectr/op_hugectr.cc) 内部的 backend 分发。

| 字段 | 类型 | 必填 | 默认值 | 说明 |
|------|------|------|--------|------|
| `backend` | string | 否 | `recstore` | HugeCTR 运行时后端，可选 `recstore` 或 `hierkv` |

当前行为：

- `recstore`：沿用现有 host staging + RecStore OP 路径
- `hierkv`：切到 HugeCTR 专用 HierKV backend 适配层

## 6. hierkv 配置

`hierkv` 顶层配置仅在 `hugectr.backend = "hierkv"` 时使用，用于初始化 HugeCTR 专用 HierarchicalKV backend。

| 字段 | 类型 | 必填 | 说明 |
|------|------|------|------|
| `max_capacity` | integer | 是 | HierarchicalKV 最大容量 |
| `max_hbm_for_vectors` | integer | 是 | 向量可使用的 HBM 上限，单位字节 |
| `dim` | integer | 是 | embedding 维度 |
| `device_id` | integer | 否 | 可选 CUDA 设备号 |
| `reserved_key_start_bit` | integer | 否 | 可选保留 key 起始 bit，范围 0-62 |

注意：

- 这组配置当前只进入 HugeCTR 侧适配层，不会回流到 `CommonOp`
- 如果选择 `hierkv` 但缺少上述必填字段，HugeCTR 入口会直接抛出配置错误

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

                "capacity": 40000000,
                "value_size": 512,

                "index": {"type": "DRAM_EXTENDIBLE_HASH"},
                "value_memory_management": "PersistLoopShmMalloc"
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

??? example "DRAM 索引 + SSD 值配置"
    ```json
    {
        "cache_ps": {
            "base_kv_config": {
                "path": "/data/recstore",
                "index": {"type": "DRAM_EXTENDIBLE_HASH"},

                "capacity": 2000000,
                "value_size": 128,
                "value_memory_management": "PersistLoopShmMalloc"
            }
        }
    }
    ```

    推导引擎类型：`KVEngineExtendibleHash`

??? example "SSD 索引 + SSD 值配置"
    ```json
    {
        "cache_ps": {
            "base_kv_config": {
                "path": "/data/recstore",
                "index": {"type": "SSD_EXTENDIBLE_HASH", "path": "/data/recstore/index.db"},

                "capacity": 2000000,
                "value_size": 128,
                "value_memory_management": "PersistLoopShmMalloc",
                "queue_size": 1024
            }
        }
    }
    ```

    推导引擎类型：`KVEngineCCEH`

??? example "HYBRID 混合模式配置"

    ```json
    {
        "cache_ps": {
            "base_kv_config": {
                "path": "/data/recstore",
                "index": {"type": "DRAM_EXTENDIBLE_HASH"},

                "shmcapacity": 10737418240,
                "ssdcapacity": 107374182400,
                "value_memory_management": "PersistLoopShmMalloc"
            }
        }
    }
    ```

    推导引擎类型：`KVEngineHybrid`

??? example "在 CI 中的配置"

    Github Actions 服务器环境限制 CPU 使用，同时无显卡支持，在 ci 配置脚本中使用：

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

    来配置DRAM 索引 + SSD 值配置。

## 配置文件使用

配置文件通常保存为 `recstore_config.json`，可通过以下方式读取：

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
