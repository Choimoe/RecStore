# 读写混合存储引擎与网络层吞吐矩阵

单位：`M keys/s`。表格数值为多轮成功实验的平均吞吐。

## 参数

- workload：读写混合，`mode=mixed`
- read ratio：`50`
- write ratio：`50`
- value size：`512 bytes`
- embedding dim：`128 float`
- key / row 数：`1,000,000`
- batch size：`1024`
- 客户端并发线程数：`16`
- 网络层 benchmark：16 个客户端 worker 线程并行压测
- 网络层 RDMA：`raw_message`，PUT-v2 `read`，`rdma_thread_num=16`
- 存储层数据：`/tmp/recstore_bench/storage_big_1m_mixed50_r123/summary.csv`
- 网络层 GRPC/BRPC 数据：
  - `/tmp/recstore_bench/network_big_1m_mixed50_grpc_brpc_r123.csv`
  - `/tmp/recstore_bench/network_big_1m_mixed50_grpc_brpc_r2.csv`
  - `/tmp/recstore_bench/network_big_1m_mixed50_grpc_brpc_r3.csv`
- 汇总矩阵：`/tmp/recstore_bench/storage_network_matrix_big_1m_mixed50.csv`

## 吞吐矩阵

| 存储引擎 | 存储层 | GRPC | BRPC | RDMA |
| --- | ---: | ---: | ---: | ---: |
| `hps` | 2.828 | N/A | N/A | N/A |
| `dram_eh_dram` | 2.949 | 2.244 | 2.601 | 失败 |
| `dram_pet_dram` | 4.431 | 2.388 | 2.739 | 失败 |
| `dram_map_dram` | 2.134 | 2.361 | 2.941 | 失败 |

## 口径说明

- 本表是读写混合结果，不复用纯读 `fetch` 数据。
- `hps` 当前只参与纯存储层 benchmark；PS 网络层 benchmark 没有 HPS 的 GRPC、BRPC、RDMA lane。
- `dram_eh_dram` 网络层对应 `DRAM_EXTENDIBLE_HASH`。
- `dram_pet_dram` 网络层对应 `DRAM_PET_HASH`。
- `dram_map_dram` 网络层对应 `DRAM_UNORDERED_MAP`。
- RDMA raw_message 在该读写混合口径下未得到有效吞吐：`DRAM_EXTENDIBLE_HASH/RDMA` 和 `DRAM_UNORDERED_MAP/RDMA` 均在 load 成功后，于 16 线程 mixed 运行阶段触发 `WaitRPCFinish timeout`；`DRAM_PET_HASH/RDMA` 的 `fetch_insert` 探针也触发同类 GET timeout。因此本表不填入旧的纯读 RDMA 结果。

失败证据：

- `/tmp/recstore_bench/network_big_1m_mixed50_r1`
- `/tmp/recstore_bench/network_big_1m_mixed50_rdma_ext_after_lock_probe`
- `/tmp/recstore_bench/network_big_1m_mixed50_rdma_map_pet_probe`
- `/tmp/recstore_bench/network_big_1m_fetch_insert_rdma_pet_probe`
