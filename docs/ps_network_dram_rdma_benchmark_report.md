# RecStore DRAM / Network / PyTorch RDMA 性能对比报告

日期：2026-05-21  
工作分支：`bench/ps-network-dram`  
数据目录：`/tmp/recstore_bench`

## 1. 目标与结论

本轮工作完成了 RDMA 在网络层 benchmark 与 `model_zoo/rs_demo` PyTorch 路径中的接入，并在同一批参数下对 DRAM 存储后端、网络层 GRPC / BRPC / RDMA、PyTorch 层 GRPC / BRPC / RDMA 做了重复实验。

主要结论：

- 纯存储层中，`dram_pet_dram` 吞吐最高，3 轮均值为 `27.440 M keys/s`；其次是 `dram_eh_dram` 的 `20.015 M keys/s`。
- 网络层中，在同一 DRAM 后端、同一 workload 下，RDMA 明显高于 GRPC / BRPC；例如 `DRAM_PET_HASH` 下 RDMA 为 `5.162 M keys/s`，BRPC 为 `0.975 M keys/s`，GRPC 为 `0.648 M keys/s`。
- PyTorch 小请求闭环中，RDMA 路径吞吐最高，3 轮均值为 `267.225 samples/s`，高于 BRPC 的 `231.930 samples/s` 和 GRPC 的 `193.878 samples/s`。
- FasterKV 当前环境不可用，`backend_benchmark --backend=fasterkv` 明确报错：`fasterkv backend is unavailable: fasterkv_backend target was not built`。因此本报告不声称 FasterKV 实测吞吐。
- PyTorch RDMA 当前完成的是单进程、单 shard、小请求规模的等价对比。更大 batch / embedding dim 下触发 RDMA raw message envelope 限制，需要继续扩展 RDMA 分片或大消息协议后再纳入等价大 batch 对比。

## 2. RDMA 接入实现摘要

### 2.1 网络层 benchmark

实现位置：

- `src/test/scripts/run_ps_dram_transport_benchmark.py`
- `src/benchmark/ps_transport_benchmark.cc`
- `src/ps/rdma/rdma_ps_client_adapter.cc`
- `src/ps/rdma/rdma_ps_client_adapter.h`

关键改动：

- `run_ps_dram_transport_benchmark.py` 增加 `RDMA` transport lane，RDMA lane 使用 `PetPSClusterRunner` 启动 `petps_server` 和 memcached，并传入 `rdma_use_dram=True`。
- RDMA runtime config 使用 `cache_ps.ps_type=RDMA`，后端仍为 `DRAM_VALUE_STORE`，value path 放在 `/dev/shm/recstore_ps_dram_bench/...`，与本轮 DRAM 网络对比保持一致。
- `ps_transport_benchmark.cc` 支持 `--workload=transactions` 并输出统一的 `PS_BENCHMARK_RESULT` 行，便于 GRPC / BRPC / RDMA 汇总到同一 CSV。
- `RDMAPSClientAdapter` 支持从环境变量读取 RDMA push slot、arena、timeout、transfer mode 等参数，并修复同步 GET receive arena 耗尽和嵌入 Python 进程时 gflags argv 污染问题。

### 2.2 PyTorch 层 RDMA

实现位置：

- `model_zoo/rs_demo/config.py`
- `model_zoo/rs_demo/cli.py`
- `model_zoo/rs_demo/runtime/server.py`
- `model_zoo/rs_demo/run_mock_stress.py`

关键改动：

- `--ps-type` 增加 `RDMA`。
- 增加 RDMA 参数：`--rdma-thread-num`、`--rdma-max-keys-per-request`、memcached 配置、transport mode、PUT protocol、push slot 大小等。
- `rs_demo` RDMA server 生命周期改为通过 `PetPSClusterRunner.run()` 管理，启动 `petps_server` 和 memcached，并把 RDMA client/server 环境变量注入当前进程。
- RDMA `rs_demo` runtime config 当前限制为单进程单 shard；value size hint 使用 `embedding_dim * 4`。

### 2.3 纯存储 benchmark 修复

实现位置：

- `tools/benchmarks/run_hps_backend_compare.py`
- `src/benchmark/backend_benchmark.cc`
- `src/benchmark/CMakeLists.txt`

关键改动：

- RecStore DRAM 后端的数据路径改为 `/dev/shm/recstore_storage_bench/...`，避免路径落到普通磁盘导致纯 DRAM 对比失真。
- FasterKV 未构建时，benchmark 明确报错而不是静默跳过或产生误导性结果。

## 3. 实验参数

### 3.1 纯存储层

数据文件：`/tmp/recstore_bench/storage_full_r123/summary.csv`

参数：

- 后端：`hps_hash_map`、`dram_eh_dram`、`dram_pet_dram`、`dram_map_dram`
- mode：`fetch`
- record count：`100000`
- runtime：`3 s`
- threads：`4`
- batch size：`64`
- value size：`512 bytes`
- distribution：`uniform`
- repeat：`3`
- RecStore DRAM value path：`/dev/shm/recstore_storage_bench/...`

FasterKV 探测文件：`/tmp/recstore_bench/fasterkv_probe.log`

### 3.2 网络层

数据文件：

- `/tmp/recstore_bench/network_full_r1.csv`
- `/tmp/recstore_bench/network_full_r2.csv`
- `/tmp/recstore_bench/network_full_r3.csv`

参数：

- index：`DRAM_EXTENDIBLE_HASH`、`DRAM_UNORDERED_MAP`、`DRAM_PET_HASH`
- value store：`DRAM_VALUE_STORE`
- transport：`GRPC`、`BRPC`、`RDMA`
- workload：`transactions`
- mode：`fetch`
- record count / capacity：`100000`
- runtime：`3 s`
- threads：`4`
- load threads：`2`
- batch size：`64`
- value size：`512 bytes`
- distribution：`uniform`
- shard：单 shard
- repeat：`3`

### 3.3 PyTorch 层

数据文件：

- 汇总：`/tmp/recstore_bench/aggregate_summary.csv`
- 原始输出：`/tmp/recstore_bench/pytorch_lanes_small/outputs/pytorch_{grpc,brpc,rdma}_r{1,2,3}/recstore_main_agg.csv`
- 日志：`/tmp/recstore_bench/pytorch_lanes_small_{grpc,brpc,rdma}_r{1,2,3}.log`

参数：

- backend：`recstore`
- ps type：`GRPC`、`BRPC`、`RDMA`
- steps：`20`
- warmup steps：`5`
- batch size：`4`
- num embeddings：`1024`
- embedding dim：`4`
- read mode：`prefetch`
- prefetch depth：`0`
- init rows：`128`
- RDMA max keys per request：`4096`
- RDMA thread num：`1`
- allocator：`PersistLoopShmMalloc`
- 数据集：`/tmp/recstore_bench/rs_demo_data/processed_day_0_data`

说明：PyTorch 层结果以 `samples/s` 和 step / lookup / update latency 为主，不与存储层或网络层的 `M keys/s` 直接等价换算。

## 4. 结果

### 4.1 纯存储层吞吐

单位：`M keys/s`，均为 3 轮 run phase。

| 后端 | 平均吞吐 | 标准差 | 3 轮结果 |
| --- | ---: | ---: | --- |
| `hps_hash_map` | 4.505 | 1.042 | 3.855 / 3.953 / 5.707 |
| `dram_eh_dram` | 20.015 | 0.038 | 19.976 / 20.018 / 20.052 |
| `dram_pet_dram` | 27.440 | 0.024 | 27.418 / 27.438 / 27.466 |
| `dram_map_dram` | 8.926 | 0.141 | 8.772 / 9.048 / 8.958 |

### 4.2 网络层吞吐

单位：`M keys/s`，均为 3 轮 run phase。

| DRAM index | GRPC 平均 | BRPC 平均 | RDMA 平均 | RDMA 标准差 |
| --- | ---: | ---: | ---: | ---: |
| `DRAM_EXTENDIBLE_HASH` | 0.662 | 0.907 | 3.718 | 1.204 |
| `DRAM_UNORDERED_MAP` | 0.631 | 0.890 | 3.797 | 1.027 |
| `DRAM_PET_HASH` | 0.648 | 0.975 | 5.162 | 1.380 |

明细：

| DRAM index | transport | 平均吞吐 | 标准差 | 3 轮结果 |
| --- | --- | ---: | ---: | --- |
| `DRAM_EXTENDIBLE_HASH` | GRPC | 0.662 | 0.020 | 0.639 / 0.673 / 0.674 |
| `DRAM_EXTENDIBLE_HASH` | BRPC | 0.907 | 0.032 | 0.942 / 0.899 / 0.879 |
| `DRAM_EXTENDIBLE_HASH` | RDMA | 3.718 | 1.204 | 3.051 / 2.996 / 5.109 |
| `DRAM_UNORDERED_MAP` | GRPC | 0.631 | 0.005 | 0.635 / 0.631 / 0.626 |
| `DRAM_UNORDERED_MAP` | BRPC | 0.890 | 0.023 | 0.909 / 0.895 / 0.864 |
| `DRAM_UNORDERED_MAP` | RDMA | 3.797 | 1.027 | 4.415 / 4.363 / 2.612 |
| `DRAM_PET_HASH` | GRPC | 0.648 | 0.009 | 0.644 / 0.641 / 0.658 |
| `DRAM_PET_HASH` | BRPC | 0.975 | 0.058 | 1.031 / 0.981 / 0.915 |
| `DRAM_PET_HASH` | RDMA | 5.162 | 1.380 | 5.979 / 5.939 / 3.570 |

### 4.3 PyTorch 层闭环结果

PyTorch 层使用小请求可稳定等价参数：batch size `4`，embedding dim `4`，steps `20`，warmup `5`。吞吐单位为 `samples/s`，延迟单位为 `ms`。

| transport | samples/s 平均 | step 平均 | lookup 平均 | update 平均 |
| --- | ---: | ---: | ---: | ---: |
| GRPC | 193.878 | 20.633 | 4.963 | 5.003 |
| BRPC | 231.930 | 17.250 | 3.817 | 2.853 |
| RDMA | 267.225 | 14.969 | 2.742 | 1.858 |

对应统一汇总表中的 `mean_mkeys_s` 字段只是按 `samples/s / 1e6` 存放，便于和统一表结构兼容；PyTorch 层分析应优先使用 `mean_samples_s`、`mean_step_ms`、`mean_lookup_ms`、`mean_update_ms`。

## 5. 等价性说明

本轮“等价”按层定义：

- 纯存储层：所有后端使用相同 record count、runtime、threads、batch size、value size、distribution 和 fetch-only workload。RecStore DRAM value store 固定在 `/dev/shm`，避免磁盘路径影响 DRAM 结果。
- 网络层：GRPC / BRPC / RDMA 使用相同 C++ `ps_transport_benchmark` workload、同一 index / value store 配置、同一 key/value 参数和 run phase 统计口径。RDMA 与 BRPC/GRPC 的差异仅在 server/client transport lane 和启动依赖上。
- PyTorch 层：GRPC / BRPC / RDMA 使用相同 `rs_demo` 数据集、模型参数、step/warmup、batch、embedding dim、prefetch 策略和 optimizer 闭环；只替换 `--ps-type` 与对应 server lifecycle。

当前不能等价比较的部分：

- FasterKV：本环境未构建 FasterKV target，无法产生有效吞吐。
- PyTorch 大请求 RDMA：batch size `64`、embedding dim `16` 曾触发 RDMA raw message envelope 限制。调整 push slot 后仍会遇到 `messeage size too large; exit -1`，因此没有纳入最终等价数据。
- RDMA PyTorch 多进程 / 多 shard：当前 `rs_demo` RDMA 路径显式限制单进程单 shard，尚未接入多 client process 的 global id 分配与多 shard 路由。

## 6. 复现命令

构建：

```bash
cmake --build build --target recstore_torch_ops ps_transport_benchmark -j$(nproc)
```

网络层单轮示例：

```bash
python3 src/test/scripts/run_ps_dram_transport_benchmark.py \
  --transports GRPC,BRPC,RDMA \
  --index-types DRAM_EXTENDIBLE_HASH,DRAM_UNORDERED_MAP,DRAM_PET_HASH \
  --num-shards 1 \
  --capacity 100000 \
  --runtime-seconds 3 \
  --threads 4 \
  --load-threads 2 \
  --batch-size 64 \
  --value-size 512 \
  --max-keys-per-request 4096 \
  --rdma-thread-num 1 \
  --rdma-put-v2-push-slot-bytes 1048576 \
  --output-dir /tmp/recstore_bench/network_full_rX_runtime \
  --csv-path /tmp/recstore_bench/network_full_rX.csv
```

纯存储层：

```bash
python3 tools/benchmarks/run_hps_backend_compare.py \
  --backends hps_hash_map dram_eh_dram dram_pet_dram dram_map_dram \
  --mode fetch \
  --record-count 100000 \
  --runtime-seconds 3 \
  --threads 4 \
  --batch-size 64 \
  --value-size 512 \
  --distribution uniform \
  --repeat 3 \
  --output-dir /tmp/recstore_bench/storage_full_r123
```

PyTorch RDMA 小请求示例：

```bash
python3 model_zoo/rs_demo/run_mock_stress.py \
  --backend recstore \
  --ps-type RDMA \
  --steps 20 \
  --warmup-steps 5 \
  --batch-size 4 \
  --num-embeddings 1024 \
  --embedding-dim 4 \
  --read-mode prefetch \
  --prefetch-depth 0 \
  --init-rows 128 \
  --rdma-max-keys-per-request 4096 \
  --rdma-thread-num 1 \
  --allocator PersistLoopShmMalloc \
  --data-dir /tmp/recstore_bench/rs_demo_data/processed_day_0_data \
  --dense-arch-layer-sizes 32,16,4 \
  --over-arch-layer-sizes 64,16,1 \
  --output-root /tmp/recstore_bench/pytorch_lanes_small/outputs \
  --run-id pytorch_rdma_rX
```

## 7. 验证记录

已运行并通过：

```bash
python3 -m unittest -v \
  src.test.scripts.test_run_ps_dram_transport_benchmark \
  src.test.scripts.test_run_rdma_transport_benchmarks \
  src.test.scripts.test_petps_cluster_runner \
  src.test.scripts.test_run_storage_backend_compare \
  model_zoo.rs_demo.tests.test_server_ports \
  model_zoo.rs_demo.tests.test_recstore_runner \
  model_zoo.rs_demo.tests.test_torchrec_config
```

结果：`Ran 124 tests ... OK`

```bash
./build/bin/test_ps_transport_benchmark
```

结果：`6 tests passed`

```bash
cmake --build build --target recstore_torch_ops ps_transport_benchmark -j$(nproc)
```

结果：构建通过。

RDMA PyTorch smoke：

```bash
python3 model_zoo/rs_demo/run_mock_stress.py \
  --backend recstore --ps-type RDMA \
  --steps 2 --warmup-steps 0 --batch-size 4 \
  --num-embeddings 1024 --embedding-dim 4 \
  --read-mode prefetch --prefetch-depth 0 --init-rows 128 \
  --rdma-max-keys-per-request 4096 --rdma-thread-num 1 \
  --allocator PersistLoopShmMalloc \
  --data-dir /tmp/recstore_bench/rs_demo_data/processed_day_0_data \
  --dense-arch-layer-sizes 32,16,4 \
  --over-arch-layer-sizes 64,16,1 \
  --output-root /tmp/recstore_bench/rs_demo_rdma_smoke \
  --run-id rdma_smoke5
```

结果：完成 2 step，并生成 `recstore_main.csv`。

最后检查：

```bash
pgrep -af 'ps_server|petps_server|memcached|run_mock_stress|ps_transport_benchmark|backend_benchmark' || true
```

结果：未发现残留 benchmark/server 进程。

## 8. 后续建议

1. 扩展 RDMA 大消息路径，解决 raw message envelope 限制后，补齐 PyTorch batch `64`、embedding dim `16` 或更接近生产参数的 RDMA 等价实验。
2. 为 PyTorch RDMA 增加多进程、多 shard client id / routing 支持，再做分布式训练闭环对比。
3. 修复或补齐 `third_party/faster/cc` 构建，重新纳入 FasterKV 纯存储层实测。
4. 网络层 RDMA 第三轮波动较大，后续可增加 repeat 到 `5` 或 `10`，并固定 CPU affinity / NUMA 绑定以降低噪声。

## 9. 2026-05-22 大请求补测

本节覆盖用户指定的大请求参数：

- value size：`512 bytes`
- embedding dim：`128 float`
- key / row 数：`1,000,000`
- batch size：`1024`
- client 压测并发：网络层 `threads=16`
- RDMA server worker：`rdma_thread_num=16`

### 9.1 本轮修复

大请求补测前发现并修复了以下 RDMA 限制：

- raw message GET 的 key payload 超过 Mayfly `4096` byte envelope 时会失败。`RDMAPSClientAdapter` 现在对 raw-message GET 自动按 `400` keys 分片，并使用 RDMA registered receive buffer，而不是 heap buffer。
- PyTorch prefetch 也会发起大 GET。现在 `PrefetchParameter` 对 raw-message 大请求记录多个 chunk RPC，并在 `GetPrefetchResultFlat` 组装回原顺序。
- PyTorch / adapter 层 PUT 初始化不能一次发送 1M 行。`RDMAPSClientAdapter::PutParameter` 现在按 `max_kv_num_per_request` 分片；Python `KVClient.init_data` 对默认零初始化按 `4096` 行分片，避免一次构造和发送 512MB tensor。
- `rs_demo` 新增大请求 RDMA 参数透传：`rdma_per_thread_response_limit_bytes`、`rdma_client_receive_arena_bytes`、`rdma_put_client_send_arena_bytes`、`rdma_put_server_scratch_bytes`、`rdma_put_v2_push_region_offset`。

### 9.2 网络层结果

命令参数固定为：

- `capacity=1000000`
- `runtime_seconds=5`
- `threads=16`
- `load_threads=1`
- `batch_size=1024`
- `value_size=512`
- `max_keys_per_request=1024`
- RDMA：`raw_message` + PUT-v2 `read`

数据文件：

- `/tmp/recstore_bench/network_big_1m_r1b.csv`
- `/tmp/recstore_bench/network_big_1m_r2_ext_rdma_retry.csv`
- `/tmp/recstore_bench/network_big_1m_r2_tail.csv`
- `/tmp/recstore_bench/network_big_1m_r3.csv`

单位：`M keys/s`。

| DRAM index | transport | 有效轮数 | 平均吞吐 | 标准差 | 明细 |
| --- | --- | ---: | ---: | ---: | --- |
| `DRAM_EXTENDIBLE_HASH` | GRPC | 2 | 2.327 | 0.007 | 2.322 / 2.331 |
| `DRAM_EXTENDIBLE_HASH` | BRPC | 2 | 3.714 | 0.066 | 3.761 / 3.668 |
| `DRAM_EXTENDIBLE_HASH` | RDMA | 3 | 6.722 | 3.077 | 3.172 / 8.611 / 8.384 |
| `DRAM_UNORDERED_MAP` | GRPC | 3 | 2.293 | 0.020 | 2.275 / 2.290 / 2.315 |
| `DRAM_UNORDERED_MAP` | BRPC | 3 | 4.268 | 0.075 | 4.284 / 4.334 / 4.186 |
| `DRAM_UNORDERED_MAP` | RDMA | 3 | 5.099 | 0.137 | 5.231 / 4.957 / 5.108 |
| `DRAM_PET_HASH` | GRPC | 3 | 2.341 | 0.050 | 2.398 / 2.309 / 2.315 |
| `DRAM_PET_HASH` | BRPC | 3 | 3.662 | 0.114 | 3.670 / 3.545 / 3.772 |
| `DRAM_PET_HASH` | RDMA | 3 | 8.399 | 0.584 | 8.275 / 7.886 / 9.035 |

说明：

- `r2` 的 `DRAM_EXTENDIBLE_HASH/RDMA` 在全矩阵运行中出现一次 `GetParameter failed`，单独重跑同参数通过，结果为 `8.611 M keys/s`。报告保留这次异常，说明该 index + RDMA lane 仍有一次性波动风险。
- 网络层结果使用 C++ benchmark 的 16 个客户端 worker 线程并行压测，避免单客户端过网压不满。

### 9.3 PyTorch 层结果

PyTorch 层命令固定为：

- `backend=recstore`
- `ps_type=GRPC/BRPC/RDMA`
- `steps=20`
- `warmup_steps=5`
- `batch_size=1024`
- `num_embeddings=1000000`
- `embedding_dim=128`
- `init_rows=1000000`
- `read_mode=prefetch`
- `prefetch_depth=0`
- `allocator=PersistLoopShmMalloc`

RDMA 额外参数：

- `rdma_max_keys_per_request=1024`
- `rdma_thread_num=16`
- `rdma_transport_mode=raw_message`
- `rdma_put_v2_transfer_mode=read`
- `rdma_client_receive_arena_bytes=536870912`
- `rdma_put_client_send_arena_bytes=67108864`
- `rdma_per_thread_response_limit_bytes=1048576`
- `rdma_put_server_scratch_bytes=1048576`

数据文件：

- `/tmp/recstore_bench/pytorch_big/outputs/grpc_big_r1/recstore_main_agg.csv`
- `/tmp/recstore_bench/pytorch_big/outputs/brpc_big_r1/recstore_main_agg.csv`
- `/tmp/recstore_bench/pytorch_rdma_big/outputs/rdma_big_r1c/recstore_main_agg.csv`

| transport | samples/s | step mean ms | emb stage ms | lookup wait ms | sparse update ms |
| --- | ---: | ---: | ---: | ---: | ---: |
| GRPC | 12522.60 | 81.772 | 37.125 | 14.925 | 38.626 |
| BRPC | 11038.86 | 92.763 | 37.207 | 15.542 | 48.887 |
| RDMA | 15448.24 | 66.286 | 25.914 | 4.434 | 33.985 |

说明：

- PyTorch 层是单进程训练闭环，不等同于网络层 16 客户端线程压满链路；它用于验证模型集成路径在同等大表、大 batch、大 dim 参数下可运行并比较端到端 step/lookup/update。
- 当前数据集 `/tmp/recstore_bench/rs_demo_data/processed_day_0_data` 只有 `2048` 条样本，20 step 会复用数据；表规模与初始化行数仍为 `1,000,000`。

## 10. 存储引擎 x 网络层矩阵

本节按用户要求整理为“行是存储引擎，列是存储层、GRPC、BRPC、RDMA”的矩阵。单位为 `M keys/s`，单元格格式为 `mean +/- stdev (n=轮数)`。

固定参数：

- `value_size=512 bytes`
- `embedding_dim=128 float`，即单 value 为 128 个 `float`
- `record_count/capacity=1000000`
- `batch_size=1024`
- `threads=16`
- 网络层 benchmark 使用 16 个客户端 worker 线程并行压测
- 网络层 RDMA 使用 `raw_message` + PUT-v2 `read`，`rdma_thread_num=16`

数据文件：

- 存储层：`/tmp/recstore_bench/storage_big_1m_r123_v2/summary.csv`
- 网络层：`/tmp/recstore_bench/network_big_1m_r1b.csv`
- 网络层：`/tmp/recstore_bench/network_big_1m_r2_ext_rdma_retry.csv`
- 网络层：`/tmp/recstore_bench/network_big_1m_r2_tail.csv`
- 网络层：`/tmp/recstore_bench/network_big_1m_r3.csv`
- 网络层：`/tmp/recstore_bench/network_big_1m_r4.csv`
- 网络层：`/tmp/recstore_bench/network_big_1m_r5.csv`
- 汇总矩阵：`/tmp/recstore_bench/storage_network_matrix_big_1m.csv`

| 存储引擎 | 存储层 | GRPC | BRPC | RDMA |
| --- | ---: | ---: | ---: | ---: |
| `hps` | 2.840 +/- 0.006 (n=3) | N/A | N/A | N/A |
| `dram_eh_dram` | 35.207 +/- 0.384 (n=3) | 2.275 +/- 0.068 (n=4) | 3.706 +/- 0.085 (n=4) | 7.476 +/- 2.410 (n=5) |
| `dram_pet_dram` | 49.461 +/- 0.019 (n=3) | 2.342 +/- 0.050 (n=5) | 3.746 +/- 0.145 (n=5) | 8.509 +/- 0.472 (n=5) |
| `dram_map_dram` | 9.166 +/- 0.297 (n=3) | 2.280 +/- 0.038 (n=5) | 4.270 +/- 0.061 (n=5) | 5.134 +/- 0.122 (n=5) |
| `fasterkv` | 不可用 | N/A | N/A | N/A |

逐轮明细：

| 存储引擎 | 存储层 runs | GRPC runs | BRPC runs | RDMA runs |
| --- | --- | --- | --- | --- |
| `hps` | 2.835 / 2.839 / 2.846 |  |  |  |
| `dram_eh_dram` | 34.765 / 35.460 / 35.396 | 2.322 / 2.331 / 2.182 / 2.264 | 3.761 / 3.668 / 3.605 / 3.790 | 3.172 / 8.611 / 8.384 / 8.738 / 8.475 |
| `dram_pet_dram` | 49.445 / 49.456 / 49.483 | 2.398 / 2.309 / 2.315 / 2.393 / 2.293 | 3.670 / 3.545 / 3.772 / 3.820 / 3.924 | 8.275 / 7.886 / 9.035 / 8.434 / 8.915 |
| `dram_map_dram` | 9.453 / 9.185 / 8.861 | 2.275 / 2.290 / 2.315 / 2.218 / 2.302 | 4.284 / 4.334 / 4.186 / 4.230 / 4.314 | 5.231 / 4.957 / 5.108 / 5.109 / 5.266 |
| `fasterkv` | 不可用：`run_hps_backend_compare.py` 当前报 `unknown backend alias 'fasterkv'` |  |  |  |

说明：

- `hps` 只参与纯存储层 benchmark；当前 PS 网络层 benchmark 没有 HPS 的 GRPC/BRPC/RDMA lane，因此网络列标为 `N/A`。
- `dram_eh_dram` 网络层对应 `DRAM_EXTENDIBLE_HASH`，`dram_pet_dram` 对应 `DRAM_PET_HASH`，`dram_map_dram` 对应 `DRAM_UNORDERED_MAP`。
- `DRAM_EXTENDIBLE_HASH/RDMA` 有一轮全矩阵运行失败后单独同参数重跑成功；本矩阵只统计成功样本，因此 RDMA 明细包含 5 个成功值，其中 `3.172 M keys/s` 是早期成功轮的低值，导致该行 RDMA 标准差较大。
- `fasterkv` 当前 benchmark 脚本不支持该 alias，没有有效可比数据，本报告不虚构结果。
