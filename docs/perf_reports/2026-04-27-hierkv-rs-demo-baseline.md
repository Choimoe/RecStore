# HierKV RsDemo Baseline Report

## 1. Goal

本轮目标不是只看 `hierkv` microbenchmark，而是验证它接到 `rs_demo` 模型层之后的真实 end-to-end 表现，并把性能拆到足够细的阶段，回答两个问题：

1. `hierkv` single-node distributed fast path 在当前代码里是否真的生效。
2. 如果生效，它在 `rs_demo` 里的总性能是赢在存储层，还是输在模型层 glue / cross-rank 重组。

同时补做一个归因层：

- 用仓库现有 `local_shm` mixed benchmark 给出本地共享内存 backend 的独立 lower bound。
- 说明为什么当前还不能把 `local_shm` 当成一个和 `hierkv` 完全公平的 `rs_demo` end-to-end baseline。

## 2. Code / Bring-up Fixes

本轮为了拿到可信结果，先修了几处会直接污染实验结论的问题。

### 2.1 Fast-path backend flags did not actually reach workers

之前 torchrun worker 没收到：

- `--enable-single-node-distributed-fast-path`
- `--single-node-ps-backend`
- `--single-node-owner-policy`

这会导致主进程以为在测 fast path，worker 实际仍走普通远程路径。

### 2.2 Local backend shard activation semantics were wrong

`ShardedRecstoreClient.activate_shard()` 之前无条件走 `set_ps_config()`。

这对 `hierkv` 和 `local_shm` 都不对：

- `hierkv` 本地 runtime 不该被重建成 RPC client。
- `local_shm` 也不该用远程 host/port 重建 client。

现在逻辑改成：

- 只有非本地 backend 才真正 `set_ps_config()`。
- `local_shm` / `hierkv` 只记录逻辑 active shard。

### 2.3 Worker runtime dir was drifting

父进程启动 server 使用一份 runtime dir，但 torchrun worker 如果没有收到固定的 `--recstore-runtime-dir`，会各自重新生成新的 runtime config。

这在 `hierkv` 下不一定立刻炸，但在 `local_shm` 下会直接让 worker 连到错误的 `region_name`。

现在 `cli.py` 会把生成后的 runtime dir 回填到 `cfg.recstore_runtime_dir`，确保父子进程使用同一份 runtime。

### 2.4 RsDemo could not start a real LOCAL_SHM server

之前 `rs_demo` 只能启动 `ps_server`，等价于只支持 `BRPC/GRPC` server bring-up。

现在 runtime/server 层补了：

- `LOCAL_SHM` runtime config 生成
- `local_shm_ps_server` 启动
- `LOCAL_SHM` ready check
- `local_shm` 专用 runtime section
- `value_size = embedding_dim * 4` 对齐

### 2.5 TorchRec fallback and perf counters

为保证当前环境无 `torchrec` 也能跑通 `rs_demo`，并支持阶段归因，补了：

- torchrec fallback 容器
- lookup / sparse update 细粒度计时
- `lookup_breakdown_ms`
- `sparse_update_breakdown_ms`

## 3. Verification

本轮重新跑过的关键测试：

- `python3 -m unittest model_zoo.rs_demo.tests.test_recstore_distributed model_zoo.rs_demo.tests.test_server_ports model_zoo.rs_demo.tests.test_torchrec_config`
- `python3 -m unittest model_zoo.rs_demo.tests.test_recstore_runner src.python.pytorch.recstore.unittest.test_embeddingbag_single_node_distributed src.python.pytorch.recstore.unittest.test_sparse_optimizer_single_node_distributed`

结果：上述测试均通过。

## 4. Experiment Setup

### 4.1 End-to-end RsDemo runs

共同配置：

- `nnodes=1`
- `nproc_per_node=2`
- `steps=8`
- `warmup_steps=2`
- `batch_size=256`
- `num_embeddings=50000`
- `embedding_dim=64`
- `dense_arch_layer_sizes=256,128,64`
- `allocator=PersistLoopShmMalloc`

真实 fast-path `hierkv`：

```bash
python3 model_zoo/rs_demo/run_mock_stress.py \
  --backend recstore \
  --nnodes 1 \
  --nproc-per-node 2 \
  --enable-single-node-distributed-fast-path \
  --single-node-ps-backend hierkv \
  --steps 8 \
  --warmup-steps 2 \
  --batch-size 256 \
  --num-embeddings 50000 \
  --embedding-dim 64 \
  --dense-arch-layer-sizes 256,128,64 \
  --allocator PersistLoopShmMalloc \
  --output-root /tmp/rs_demo_perf \
  --run-id hierkv-report1
```

对照 `brpc` 远程路径：

```bash
python3 model_zoo/rs_demo/run_mock_stress.py \
  --backend recstore \
  --nnodes 1 \
  --nproc-per-node 2 \
  --steps 8 \
  --warmup-steps 2 \
  --batch-size 256 \
  --num-embeddings 50000 \
  --embedding-dim 64 \
  --dense-arch-layer-sizes 256,128,64 \
  --allocator PersistLoopShmMalloc \
  --output-root /tmp/rs_demo_perf \
  --run-id brpc-report1
```

### 4.2 Local SHM lower-bound benchmark

可稳定跑通的 cross-process standalone benchmark：

```bash
python3 src/test/scripts/run_local_shm_mixed_benchmark.py \
  --repo-root /app/RecStore \
  --benchmark-binary /app/RecStore/build/bin/recstore_mixed_benchmark \
  --server-binary /app/RecStore/build/bin/local_shm_ps_server \
  --iterations 8 \
  --rounds 1 \
  --warmup-rounds 0 \
  --batch-keys 128 \
  --embedding-dim 64 \
  --num-embeddings 4096 \
  --startup-delay 0.5
```

## 5. Main Results

### 5.1 HierKV fast path is real

`/tmp/rs_demo_perf/outputs/hierkv-report1/recstore_worker_rank0.log` 中有：

- `enabled=True`
- `mode=single_node`
- `backend=hierkv`
- `can_use=True`

同时 non-warmup step 中：

- `lookup_wait_ms = 0`
- `lookup_fallback_pull_ms = 0`
- `lookup_local_lookup_ms > 0`
- `update_local_apply_ms > 0`

这说明它不是伪 fast path，而是真的绕开了普通 RPC pull/wait 路径。

### 5.2 End-to-end result: current HierKV path is slower than BRPC baseline

按非 warmup step 统计：

| Metric | HierKV mean | HierKV median | BRPC mean | BRPC median |
| --- | ---: | ---: | ---: | ---: |
| `step_total_ms` | 424.63 | 416.87 | 202.30 | 209.31 |
| `lookup_breakdown_ms` | 237.21 | 223.55 | 12.89 | 13.43 |
| `sparse_update_breakdown_ms` | 4.70 | 4.45 | 18.63 | 19.56 |

直接结论：

- 当前 `hierkv` end-to-end **总 step 时间约为 `brpc` 的 1.99x**。
- 但 `hierkv` sparse update **比 `brpc` 快约 4.39x**。
- 总体输掉的原因不在 update，而在 lookup path。

### 5.3 HierKV lookup bottleneck is not storage, but reassemble

`hierkv-report1` 非 warmup step 统计：

| Metric | Mean (ms) | Median (ms) |
| --- | ---: | ---: |
| `lookup_owner_exchange_ms` | 2.68 | 1.92 |
| `lookup_local_lookup_ms` | 2.16 | 1.64 |
| `lookup_reassemble_ms` | 232.37 | 216.44 |

关键观察：

- 真正的本地 KV lookup 只占 `~1.6ms` median。
- owner exchange 也只在 `~1.9ms` median。
- **绝大部分 lookup 时间都耗在 `reassemble`，median `216ms+`。**

也就是说，在当前 `rs_demo` 集成形态下：

- `hierkv` backend 本身不是主瓶颈。
- 主瓶颈是模型层 single-node distributed glue，尤其是 embedding response 的跨 rank 重组。

### 5.4 HierKV sparse update path is already good

`hierkv-report1` 非 warmup step：

| Metric | Mean (ms) | Median (ms) |
| --- | ---: | ---: |
| `update_trace_merge_ms` | 0.39 | 0.36 |
| `update_owner_exchange_ms` | 2.32 | 2.29 |
| `update_local_apply_ms` | 1.99 | 1.91 |
| `sparse_update_breakdown_ms` | 4.70 | 4.45 |

这条链路已经比较健康：

- update trace merge 很小
- owner exchange 可控
- local apply 也不大

所以如果后续要继续优化，优先级应该明显偏向 lookup side，而不是 sparse update side。

## 6. Local SHM Attribution Layer

### 6.1 Standalone local_shm benchmark proves the transport is fast

`recstore_mixed_benchmark` 在 `LOCAL_SHM` 下的 standalone 结果：

| Phase | Config | Mean |
| --- | --- | ---: |
| init | `num_embeddings=4096, embedding_dim=64` | `419765us` |
| measure | `batch_keys=128, iterations=8` | `6209us` |

对应吞吐：

- `ops_per_sec = 2576.9`
- `key_ops_per_sec = 329844`

这说明：

- `local_shm` 共享内存 transport 自身不是完全不可用。
- 它可以作为一个“本地 backend lower bound”的归因参考。

### 6.2 But local_shm is still not a fair RsDemo end-to-end baseline

虽然 standalone local_shm benchmark 可跑，但 `rs_demo` 模型层接入下，`LOCAL_SHM` 仍不稳定：

1. 早期失败是 server/runtime 配置问题：
   - backend 被 `set_ps_config()` 覆盖
   - worker runtime dir 漂移
   - `region_name` 不一致
   - `value_size` 不匹配

2. 修完以上集成问题后，`rs_demo` 路径仍然失败：
   - direct Python `RecStoreClient + LOCAL_SHM server` 最小闭环里 `init_embedding_table` 失败
   - `rs_demo` 多进程路径里会在 table init / write 阶段失败或崩溃

因此当前结论必须收敛为：

- `local_shm` 可作为 **独立 lower bound attribution**。
- 但它 **还不能** 作为和 `hierkv` 完全公平的 `rs_demo` model-layer end-to-end baseline。

## 7. Why HierKV Loses End-to-End Today

从这轮结果看，`hierkv` 当前地位更像：

- **存储层很快**
- **模型层 glue 很重**

所以 end-to-end 输掉的根本原因不是 “HierKV backend 慢”，而是：

1. owner rank 收到本地 lookup 结果之后，跨 rank response reassemble 太重。
2. 现在的 single-node distributed path 仍然在 Python / tensor packaging / cross-rank exchange 上有高额成本。
3. lookup 这条路径的 glue 开销远大于 backend lookup 本体。

换句话说：

- `hierkv` 释放的是 “local lookup / local update” 的 backend latency。
- 但当前 `rs_demo` 架构还没把这部分收益保住，尤其在 lookup response fan-in / reassemble 上被重新吃掉了。

## 8. Actionable Next Steps

### 8.1 Highest-priority optimization

优先优化：

- `src/python/pytorch/torchrec_kv/EmbeddingBag.py`
- `src/python/pytorch/recstore/single_node_exchange.py`

目标：

- 减少 lookup response 的 Python-side list/tensor 重建
- 降低 rank 间 response reassemble 成本
- 避免重复 materialize embedding payload

### 8.2 Fair local_shm baseline unblock

继续排查：

- `src/framework/op.cc`
- `src/framework/pytorch/op_torch.cc`
- `src/python/pytorch/recstore/KVClient.py`

目标：

- 找出为什么 `LOCAL_SHM` 在 standalone benchmark 可用，但在 PyTorch op / `RecStoreClient` 路径下仍会在 init/write 阶段失败。

### 8.3 Reporting discipline

在 `local_shm` model-layer path 真正跑通之前，后续报告里应该严格区分：

- `hierkv` end-to-end rs_demo result
- `brpc` end-to-end control result
- `local_shm` standalone lower-bound attribution

不能把第三类结果包装成完全公平的 end-to-end baseline。

## 9. Artifacts

主要输出目录：

- `hierkv-report1`
  - `/tmp/rs_demo_perf/outputs/hierkv-report1/recstore_main.csv`
  - `/tmp/rs_demo_perf/outputs/hierkv-report1/recstore_main_agg.csv`
  - `/tmp/rs_demo_perf/outputs/hierkv-report1/recstore_worker_rank0.log`
- `brpc-report1`
  - `/tmp/rs_demo_perf/outputs/brpc-report1/recstore_main.csv`
  - `/tmp/rs_demo_perf/outputs/brpc-report1/recstore_main_agg.csv`
  - `/tmp/rs_demo_perf/outputs/brpc-report1/recstore_embupdate.csv`
- standalone local_shm mixed benchmark
  - runtime dir 由脚本打印，例如 `/tmp/recstore_local_shm_bench_r03o7pus`

## 10. Final Verdict

截至本轮：

- `hierkv` 已经真实接入 `rs_demo` 模型层，并能跑出可信的 single-node distributed fast path。
- 当前 end-to-end 性能 **没有赢过** `brpc` 对照，主要输在 lookup response reassemble，而不是输在 KV backend 本体。
- `hierkv` sparse update path 已经明显优于 `brpc`。
- `local_shm` 目前还只能作为 lower-bound attribution，不能作为公平的 `rs_demo` baseline。

因此下一步最值得做的不是再改 `hierkv` backend，而是继续压缩模型层 lookup glue，尤其是 response reassemble。
