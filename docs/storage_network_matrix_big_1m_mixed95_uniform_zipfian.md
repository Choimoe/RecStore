# Uniform / Zipfian 读 95 写 5 吞吐矩阵

单位：`M keys/s`。表格数值为多轮成功实验的平均吞吐。

## 参数

- workload：读写混合，`mode=mixed`
- read ratio：`95`
- write ratio：`5`
- value size：`512 bytes`
- embedding dim：`128 float`
- key / row 数：`1,000,000`
- batch size：`1024`
- 客户端并发线程数：`16`
- 网络层 benchmark：16 个客户端 worker 线程并行压测
- zipfian alpha：`0.9`
- 网络层 RDMA：`raw_message`，PUT-v2 `read`，`rdma_thread_num=16`
- 汇总矩阵：`/tmp/recstore_bench/storage_network_matrix_big_1m_mixed95_uniform_zipfian.csv`

## Uniform

| 存储引擎 | 存储层 | GRPC | BRPC | RDMA |
| --- | ---: | ---: | ---: | ---: |
| `hps` | 2.560 | N/A | N/A | N/A |
| `dram_eh_dram` | 7.884 | 2.455 | 2.963 | 失败 |
| `dram_pet_dram` | 10.276 | 2.336 | 2.840 | 失败 |
| `dram_map_dram` | 3.521 | 2.400 | 3.774 | 失败 |

## Zipfian

| 存储引擎 | 存储层 | GRPC | BRPC | RDMA |
| --- | ---: | ---: | ---: | ---: |
| `hps` | 3.282 | N/A | N/A | N/A |
| `dram_eh_dram` | 9.144 | 2.369 | 2.836 | 失败 |
| `dram_pet_dram` | 13.189 | 2.467 | 2.734 | 失败 |
| `dram_map_dram` | 2.870 | 2.395 | 3.762 | 失败 |

## 数据文件

Uniform：

- 存储层：`/tmp/recstore_bench/storage_big_1m_mixed95_uniform_r123/summary.csv`
- 网络层：
  - `/tmp/recstore_bench/network_big_1m_mixed95_uniform_grpc_brpc_r1.csv`
  - `/tmp/recstore_bench/network_big_1m_mixed95_uniform_grpc_brpc_r2.csv`
  - `/tmp/recstore_bench/network_big_1m_mixed95_uniform_grpc_brpc_r3.csv`

Zipfian：

- 存储层：`/tmp/recstore_bench/storage_big_1m_mixed95_zipfian_r123/summary.csv`
- 网络层：
  - `/tmp/recstore_bench/network_big_1m_mixed95_zipfian_grpc_brpc_r1.csv`
  - `/tmp/recstore_bench/network_big_1m_mixed95_zipfian_grpc_brpc_r2.csv`
  - `/tmp/recstore_bench/network_big_1m_mixed95_zipfian_grpc_brpc_r3.csv`

## 口径说明

- 本文档使用读 95 写 5 的混合负载，不复用纯读 `fetch` 或读写 50/50 数据。
- `hps` 当前只参与纯存储层 benchmark；PS 网络层 benchmark 没有 HPS 的 GRPC、BRPC、RDMA lane。
- `dram_eh_dram` 网络层对应 `DRAM_EXTENDIBLE_HASH`。
- `dram_pet_dram` 网络层对应 `DRAM_PET_HASH`。
- `dram_map_dram` 网络层对应 `DRAM_UNORDERED_MAP`。
- RDMA raw_message 在 uniform 和 zipfian 的读 95 写 5 口径下未得到有效吞吐：uniform 在 load 成功后于 16 线程 mixed 运行阶段触发 `WaitRPCFinish timeout`；zipfian 在 RDMA 探针中出现内存错误。因此 RDMA 列标为失败，不填入旧的纯读 RDMA 数据。

RDMA 失败证据：

- uniform：`/tmp/recstore_bench/network_big_1m_mixed95_uniform_rdma_pet_probe`
- zipfian：`/tmp/recstore_bench/network_big_1m_mixed95_zipfian_rdma_pet_probe`
