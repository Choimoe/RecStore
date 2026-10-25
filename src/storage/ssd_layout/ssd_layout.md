# 静态 SSD 布局

`recstore_ssd_layout` 根据查询日志离线计算 embedding 在 SSD 页上的固定位置，
生成 layout artifact。运行时由 `SSD_STATIC_LAYOUT_VALUE_STORE` 加载，
同一次查询里经常一起出现的 key 会落在同一页，批量读取需要的页数因此更少。

## 目录

- [概览](#概览)
- [快速开始](#快速开始)
- [策略](#策略)
- [运行时配置](#运行时配置)
- [CLI 参考](#cli-参考)
- [算法](#算法)
- [输入格式](#输入格式)
- [Artifact 格式](#artifact-格式)
- [限制](#限制)

## 概览

```text
query log ──partition──▶ bandana.bin ──replicate──▶ maxembed.bin ──shrink──▶ maxembed-small.bin
                              │                          │                         │
                              └──────────────┬───────────┴─────────────────────────┘
                                             ▼
                              SSD_STATIC_LAYOUT_VALUE_STORE
```

- **页（page）**：SSD IO 单位，大小为 `page_size`，可放
  `C = floor(page_size / row_size)` 行。
- **location**：`(page_id, byte_offset)`，一行在 SSD 上的位置。
- **primary**：每个 key 的第一个 location。单条读取只读 primary。
- **副本（replica）**：MaxEmbed 额外生成的 location。写入时所有副本同步更新。

## 快速开始

以 128 字节 embedding、4 KiB 页为例：

```bash
# Bandana: one location per key
recstore_ssd_layout partition --query query.bin --binary \
  --row-size 128 --output bandana.bin

# MaxEmbed: add up to 1.0 * key_count replicas
recstore_ssd_layout replicate --query query.bin --binary \
  --mapping bandana.bin --rep-ratio 1.0 --output maxembed.bin

recstore_ssd_layout inspect --mapping maxembed.bin
# strategy=maxembed page_size=4096 row_size=128 keys=... pages=... copies=... checksum=...
```

然后把 artifact 配进运行时，见[运行时配置](#运行时配置)。

## 策略

| strategy | 生成命令 | 每个 key 的 location | 批量读取 |
|----------|----------|----------------------|----------|
| `bandana` | `partition` | 1 | 按 primary 页聚合 |
| `maxembed` | `partition` + `replicate`，可选 `shrink` | 1 个 primary + 0..n 个副本 | 每轮选覆盖最多未读 key 的页 |

MaxEmbed 读得更少，代价是 SSD 空间和写放大（每次写入要更新全部副本）。
`plain` 是普通的 `SSD_VALUE_STORE`，不使用 artifact。

## 运行时配置

```json
"value": {
  "type": "SSD_STATIC_LAYOUT_VALUE_STORE",
  "path": "/data/recstore/value-pages.db",
  "default_value_size_hint": 128,
  "io": { "type": "IOURING", "queue_depth": 512, "base_offset_bytes": 0 },
  "ssd_layout": {
    "strategy": "maxembed",
    "mapping_path": "/data/recstore/maxembed.bin"
  }
}
```

加载时以下字段必须与 artifact 一致，否则构造失败：

| 配置 | artifact 字段 |
|------|---------------|
| `ssd_layout.strategy` | `strategy` |
| `default_value_size_hint` | `row_size` |
| 编译期 `PAGE_SIZE` | `page_size` |

`io.base_offset_bytes` 必须按页对齐，物理页号为
`page_id + base_offset_bytes / PAGE_SIZE`。未提供 `value.io` 时读取
`value.ssd_allocator.io`；两处都省略时使用 `IOURING`、queue depth 512 和
base offset 0。

在 `TIERED_VALUE_STORE` 中，给 `value` 加上同样的 `ssd_layout` 即可让 SSD
子层使用静态布局，详见
[composite_kvengine.md](../../../docs/storage/composite_kvengine.md#静态-ssd-布局)。

**读写语义**

- 只能写 artifact 中已有的 key，value 长度必须等于 `row_size`。
- `Read` 读 primary；`BatchRead` 贪心选页，MaxEmbed 可命中副本页。
- `Write` 对每个 location 所在页做 read-modify-write。
- `Alloc` 抛异常，`Free` 为空操作。

## CLI 参考

```text
recstore_ssd_layout partition --query PATH --output PATH --row-size N
                              [--page-size N] [--keys PATH] [--binary]
recstore_ssd_layout replicate --query PATH --mapping PATH --output PATH
                              [--rep-ratio F] [--keys PATH] [--binary]
recstore_ssd_layout shrink    --mapping PATH --output PATH [--rep-ratio F]
recstore_ssd_layout inspect   --mapping PATH
```

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--query` | — | query log 路径 |
| `--binary` | 文本 | 按[二进制 CSR](#二进制-query-log) 读取 |
| `--keys` | `key = vertex` | [key map](#key-map) 路径 |
| `--mapping` | — | 输入 artifact |
| `--output` | — | 输出 artifact |
| `--row-size` | — | 行字节数，`1 <= row_size <= page_size` |
| `--page-size` | `4096` | 页字节数 |
| `--rep-ratio` | `0` | 额外副本比例，有限且 `>= 0` |

`replicate` 必须使用与 `partition` 相同的 query log 和 key map。

`inspect` 输出中 `copies` 是所有 location 的总数（含 primary）。

退出码：`0` 成功；`1` 输入或校验失败（stderr 为
`recstore_ssd_layout: <reason>`）；`2` 参数错误。

## 算法

### partition

1. 出现在 query log 中的顶点为 hot，其余为 cold。
2. hot 顶点用固定种子打乱，再递归二分；每次二分做最多 8 轮基于 gain 的交换，
   让同一条查询中的顶点尽量留在同侧。区间不超过 `C` 时成为一页。
3. cold 顶点按顶点号顺序填满后续页。
4. 页内按 key 升序分配槽位，`byte_offset = slot * row_size`。

少于 2 个顶点的查询不影响划分。相同输入总是得到相同输出。

### replicate

额外副本预算 `B = ceil(rep_ratio * key_count)`。

1. 给每个顶点打分：对它参与的每条查询，累加“该查询跨越的 primary 页数 - 1”。
2. 按分数降序选中心顶点；每个中心生成一页副本，放入中心和与它共现、
   但 primary 不在同一页的邻居（按共现次数降序），最多 `min(C, 剩余预算)` 个。
3. 不足 2 个 key 的副本页跳过。

新增副本数 `<= B`，可能更少。primary 不变。

### shrink

目标 location 总数 `T = ceil(key_count * (1 + rep_ratio))`。从最大的 `page_id`
开始整页删除副本页，直到总数 `<= T`。primary 页永不删除，剩余页重新连续编号。
`--rep-ratio 0` 得到等价于 Bandana 的布局（strategy 仍是 `maxembed`）。

## 输入格式

query log 是一张超图：顶点对应 key，每条超边对应一次批量查询。

### 文本 query log

```text
<starting> <vertices> <queries> <declared_pins>
<id> <id> ...        # query 0
<id> <id> ...        # query 1
...
```

- 顶点号 = `id - starting`，必须在 `[0, vertices)`。
- header 之后至少要有 `queries` 行，空行是空查询；多出的内容忽略。
- 行内重复 id 去重；去重后的 pin 总数不能超过 `declared_pins`。

示例（4 个顶点、3 条查询）：

```text
0 4 3 6
0 1
2 3
1 2
```

### 二进制 query log

```text
uint64  queries
uint64  vertices
uint64  xpins[queries + 1]   // xpins[0] == 0, non-decreasing
uint32  pins[xpins[queries]] // each < vertices
```

第 `i` 条查询是 `pins[xpins[i] .. xpins[i+1])`。二进制格式不去重，生成方应保证
同一查询内顶点唯一。上面的文本示例对应
`xpins = {0, 2, 4, 6}`、`pins = {0, 1, 2, 3, 1, 2}`。

### Key map

空白分隔的 `vertices` 个 `uint64`，第 `i` 个是顶点 `i` 的 key。key 互不相同，
取值 `[0, 2^63 - 2]`。

## Artifact 格式

所有整数按主机字节序（x86_64 为小端）紧密写入，无 padding。

```text
Header      48 bytes
KeyRecord   × key_count    // key 严格递增
PageRecord  × page_count   // 下标即 page_id
uint64      checksum       // 之前所有字节的 FNV-1a 64
```

**Header**

| 偏移 | 类型 | 字段 |
|------|------|------|
| 0 | `char[8]` | magic = `RSLAY001` |
| 8 | `uint32` | version = `1` |
| 12 | `uint32` | strategy：`0` plain，`1` bandana，`2` maxembed |
| 16 | `uint32` | page_size |
| 20 | `uint32` | row_size |
| 24 | `uint64` | key_count |
| 32 | `uint64` | page_count |
| 40 | `uint64` | copy_count（location 总数） |

**KeyRecord**

```text
uint64  key
uint32  n                              // >= 1
{ uint64 page_id; uint32 byte_offset } × n   // 第一个是 primary
```

**PageRecord**

```text
uint32  n                              // 1 <= n <= C
{ uint64 key; uint32 byte_offset } × n       // 按 key 升序
```

`LoadArtifact` 会校验：checksum、magic、version、strategy、`copy_count`、
无尾随字节、offset 对齐且在页内、槽位不重复、页不空不超容量，以及
KeyRecord 与 PageRecord 描述的槽位完全一致。

## 限制

- key 集合和 `row_size` 在离线阶段固定，运行时不能新增 key 或写变长 value。
- 更换 artifact 需要重新 `BulkLoad`，或加载 layout identity（strategy +
  checksum）匹配的 checkpoint。
- 顶点数和超边数上限为 `2^32 - 1`；划分需要 `O(vertices + pins)` 内存。
- artifact 不做字节序转换，不能跨大小端机器使用。
