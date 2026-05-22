# 存储引擎与网络层吞吐矩阵

单位：`M keys/s`。表格数值为多轮成功实验的平均吞吐。

## 参数

- value size：`512 bytes`
- embedding dim：`128 float`
- key / row 数：`1,000,000`
- batch size：`1024`
- 客户端并发线程数：`16`
- 网络层 benchmark：16 个客户端 worker 线程并行压测
- 网络层 RDMA：`raw_message`，PUT-v2 `read`，`rdma_thread_num=16`
- 存储层数据：`/tmp/recstore_bench/storage_big_1m_r123_v2/summary.csv`
- 网络层数据：
  - `/tmp/recstore_bench/network_big_1m_r1b.csv`
  - `/tmp/recstore_bench/network_big_1m_r2_ext_rdma_retry.csv`
  - `/tmp/recstore_bench/network_big_1m_r2_tail.csv`
  - `/tmp/recstore_bench/network_big_1m_r3.csv`
  - `/tmp/recstore_bench/network_big_1m_r4.csv`
  - `/tmp/recstore_bench/network_big_1m_r5.csv`

## 吞吐矩阵

| 存储引擎 | 存储层 | GRPC | BRPC | RDMA |
| --- | ---: | ---: | ---: | ---: |
| `hps` | 2.840 | N/A | N/A | N/A |
| `dram_eh_dram` | 35.207 | 2.275 | 3.706 | 7.476 |
| `dram_pet_dram` | 49.461 | 2.342 | 3.746 | 8.509 |
| `dram_map_dram` | 9.166 | 2.280 | 4.270 | 5.134 |

## 口径说明

- `hps` 当前只参与纯存储层 benchmark；PS 网络层 benchmark 没有 HPS 的 GRPC、BRPC、RDMA lane。
- `dram_eh_dram` 网络层对应 `DRAM_EXTENDIBLE_HASH`。
- `dram_pet_dram` 网络层对应 `DRAM_PET_HASH`。
- `dram_map_dram` 网络层对应 `DRAM_UNORDERED_MAP`。
