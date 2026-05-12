# KV 引擎实现

## 概述

RecStore 提供了 5 种 KV 引擎实现，分别适用于不同的存储配置。所有引擎均继承自 `BaseKV`。

## 引擎列表

| 引擎名称 | 索引位置 | 值位置 | 文件 |
|---------|---------|--------|------|
| KVEngineMap | DRAM | DRAM | engine_map.h |
| KVEngineExtendibleHash | DRAM | SSD | engine_extendible_hash.h |
| KVEngineCCEH | SSD | SSD | engine_cceh.h |
| KVEngineHybrid | DRAM/SSD | HYBRID (DRAM+SSD) | engine_hybridkv.h |
| KVEnginePetKV | NVM | NVM | engine_pethash.h |

## 各引擎详细说明

### KVEngineMap

最简单的实现，使用 `std::unordered_map<uint64_t, std::string>` 存储键值对。

**特点**

- 纯内存，无持久化
- 单线程读写通过 mutex 保护
- 适用于小规模测试

??? example "配置示例"
    ```json
    {
        "index_type": "DRAM",
        "value_type": "DRAM",
        "capacity": 10000,
        "value_size": 128
    }
    ```

### KVEngineExtendibleHash

**特点**

- 索引在 DRAM (ExtendibleHashMap)
- 值在 SSD 持久化
- 使用可扩展哈希实现动态扩容
- 支持批量操作

**核心组件**

- `ExtendibleHashMap<uint64_t, SSDPointer>` - DRAM 索引
- `SSDValueWriter` - SSD 写入器
- `value_file_` - 值文件路径

??? example "配置示例"
    ```json
    {
        "index": {"type": "DRAM_EXTENDIBLE_HASH"},
        "value": {"type": "SSD_VALUE_STORE", "path": "/data/recstore/value.db",
        "capacity": 1000000,
        "value_size": 128
    }
    ```

### KVEngineCCEH

CCEH (Cacheline-Conscious Extendible Hash) 全部使用 SSD 存储。

**特点**

- 索引在 SSD (CCEHDirectory + CCEHSegment)
- 值在 SSD
- 缓存行感知设计，提高访问效率
- 支持分段锁，提高并发性

**核心组件**

- `CCEHDirectory` - SSD 目录结构
- `CCEHSegment[]` - SSD 段数组
- 每个 segment 对应一个 lock

??? example "配置示例"
    ```json
    {
        "index": {"type": "SSD_EXTENDIBLE_HASH", "path": "/data/recstore/index.db"},
        "value": {"type": "SSD_VALUE_STORE", "path": "/data/recstore/value.db",
        "capacity": 1000000,
        "value_size": 128
    }
    ```

### KVEngineHybrid

**特点**

- 索引可配置为 DRAM 或 SSD
- 值使用两层存储: DRAM (热数据) + SSD (冷数据)
- 自动进行冷热数据迁移
- 支持可配置的容量分配

**核心组件**

- `ValueManager valm` - 混合值管理器
  - `shmcapacity` - DRAM 层字节数
  - `ssdcapacity` - SSD 层字节数
- `Index*` - 索引指针，支持多种实现

??? example "配置示例"
    ```json
    {
        "index": {"type": "DRAM_EXTENDIBLE_HASH"},
        "value": {"type": "TIERED_VALUE_STORE",
        "shmcapacity": 10737418240,
        "ssdcapacity": 107374182400
    }
    ```

**数据流程**
1. 写入: 优先写入 DRAM 层
2. DRAM 满: 淘汰冷数据到 SSD 层
3. 读取: 先查 DRAM，未命中再查 SSD

### KVEnginePetKV

基于持久化内存 (Persistent Memory) 的实现。

**特点**

- 使用 PetMultiKV 作为底层存储
- 支持多个 shard，降低锁竞争
- 支持 RDMA 内存注册
- 提供预取优化

**核心组件**

- `PetMultiKV* shm_kv` - 分片 KV 存储
- `shard_num = 16` - 固定 16 个分片

??? example "配置示例"
    ```json
    {
        "path": "/mnt/pmem/recstore/value",
        "capacity": 1000000,
        "value_size": 128
    }
    ```

**预取方法** (通过 FLAGS_prefetch_method 控制)
- `0`: 逐个 Get
- `1`: 使用 BatchGet 预取

## 线程安全

各引擎的线程安全实现：

| 引擎 | 同步机制 |
|------|---------|
| Map | std::mutex |
| ExtendibleHash | 无锁 (依赖内存管理器的 tid) |
| CCEH | 分段锁 (per-segment lock) |
| Hybrid | std::shared_mutex |
| PetKV | 内部分片锁 |

## 工厂注册

所有引擎通过宏注册到工厂：

```cpp
FACTORY_REGISTER(BaseKV, KVEngineMap, KVEngineMap, const BaseKVConfig&);
FACTORY_REGISTER(BaseKV, KVEngineExtendibleHash, ...);
FACTORY_REGISTER(BaseKV, KVEngineCCEH, ...);
FACTORY_REGISTER(BaseKV, KVEngineHybrid, ...);
FACTORY_REGISTER(BaseKV, KVEnginePetKV, ...);
```

使用 `#include "storage/kv_engine/engine_factory.h"` 即可完成所有引擎的注册。
