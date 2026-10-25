4b89557594bf3098bd50d8b50c1c56f1c94f63f1
set-zw04-mlp-codelab-pc160.mt

## Workload 说明

本次测试比较 `bagpipe_planned` 在不同 `optimization_lookahead` 下的端到端性能。模型为
dlrm，transport 为 RDMA，client 部署为 distributed，PS 部署为 sharded-ps。client 为
`10.164.49.97/gpu0/rank0/nproc1` 和 `10.164.78.98/gpu0/rank1/nproc1`，其中远端通过
`quanta_commA:8417` 登录；PS 为 `10.164.49.97:15000/shard0` 和
`10.164.78.98:15000/shard1`。共 2 个 shard，每个 PS 一个 shard，`hash_method=city_hash`，
`max_keys_per_request=65536`。dataset 为
`/home/hadoop-quanta/VSCodeProjects/RecStore_shanhongqi/model_zoo/torchrec_dlrm/processed_day_0_data`，
output 为 `/home/hadoop-quanta/VSCodeProjects/RecStore_shanhongqi/results/lookahead_e2e_10251621`。

固定参数为 batch_size=1024，embedding_dim=128，num_embeddings=200000，steps=320，
warmup_steps=20，repeat=3，read_mode=bagpipe，prefetch_depth=0，index_type=DRAM_PET_HASH，
optimization plugin=`bagpipe_planned`，prescan=true，CUDA graphs=true，
sparse_collective=all_gather，cleanup_proportion=0.25，cache_capacity=160000。TorchRec
baseline 关闭。每机 PS threads=2、CPU threads=1、loader workers=2，并使用 continuous
loader。每轮测量 300 个有效 step，共 607232 个全局样本。

| 档位 | optimization_lookahead | prefetch_buffer_depth |
| --- | ---: | ---: |
| la1 | 1 | 0 |
| la2 | 2 | 1 |
| la4 | 4 | 3 |
| la8 | 8 | 7 |
| la16 | 16 | 15 |

在 planned BagPipe 中，lookahead 只控制未来 batch 的 CPU 侧准备队列深度，对应
`prefetch_buffer_depth=max(0, lookahead-1)`。resident lookup 已直接从 GPU 上的
`_values` 读取，不会按 lookahead 发起 PS 预取；lookup plan、梯度计算、all_gather
和 SGD 更新语义均保持不变。五档按 la1、la2、la4、la8、la16 的固定顺序串行运行，每档
3 轮，15 次运行全部成功，无 CSV 校验错误。由于未随机化运行顺序，以下差异需按这一
限制解读。

精度一致性使用
`.agents/skills/rs-loss-aligned/scripts/compare_loss.py` 校验。将 la1、la2、la8、la16
的每一轮与 la4 同轮比较，均为 `compared=640` 且 `aligned`。除 la8 第 3 轮出现
`max_abs=5.96046448e-08`、`max_rel=8.6035345e-08` 的 FP32 ULP 级波动外，其余比较的
`max_abs` 均为 0。详细结果见 `analysis/loss_compare.txt`。

## E2E 吞吐（samples/s，warmup 已剔除）

主验收口径为 `measured_global_samples / max_rank(sum(step_wall_ms))`，包含 barrier 和
每步收尾，能反映训练端到端的实际吞吐。均值和 CV 基于 3 轮重复；CV 使用总体标准差。

| lookahead | samples/s 各次 | 均值 | CV | 相对 la4 |
| ---: | --- | ---: | ---: | ---: |
| 1 | 443.314K, 449.295K, 436.631K | 443.080K | 1.17% | +0.89% |
| 2 | 438.678K, 409.641K, 449.218K | 432.512K | 3.87% | -1.52% |
| 4 | 451.212K, 438.264K, 428.052K | 439.176K | 2.16% | 基准 |
| 8 | 448.281K, 408.626K, 422.916K | 426.608K | 3.84% | -2.86% |
| 16 | 433.192K, 430.782K, 435.348K | 433.107K | 0.43% | -1.38% |

辅助口径取自各档 `summary_e2e.csv` 的 `samples_per_sec`，先按 step 汇总两个 rank 的
job samples/s，再对 300 个有效 step 和 3 轮取均值；该口径不含端到端 barrier 和每步
收尾开销。

| lookahead | samples/s 各次 | 均值 | CV | 相对 la4 |
| ---: | --- | ---: | ---: | ---: |
| 1 | 478.134K, 481.945K, 471.983K | 477.354K | 0.86% | +0.30% |
| 2 | 462.032K, 457.583K, 482.105K | 467.240K | 2.28% | -1.82% |
| 4 | 480.264K, 477.708K, 469.750K | 475.907K | 0.94% | 基准 |
| 8 | 472.340K, 427.998K, 460.254K | 453.531K | 4.13% | -4.70% |
| 16 | 468.188K, 465.475K, 469.215K | 467.626K | 0.34% | -1.74% |

两种口径都显示 lookahead 没有单调收益。la1 的主验收均值最高，但只比 la4 高 0.89%，
小于多组配置 2% 至 4% 的轮间 CV；la8 最低，约比 la4 低 2.9%，la16 约低 1.4%。这些
差异不足以支持“更深预取更快”或“la1 稳定优于 la4”的结论。更合理的表述是：在本负载
和本轮样本量下，lookahead 对吞吐没有可辨识的正收益，较深队列也没有带来加速。

## E2E 延迟分解（ms，warmup 已剔除，跨 rank 与 repeat 取均值）

| lookahead | batch_prepare | input_pack | embed_lookup | sparse_optimizer | update_overlap_prepare | step_total | p95 step_total |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 0.055 | 0.164 | 0.252 | 0.405 | 0.386 | 4.322 | 4.681 |
| 2 | 0.059 | 0.176 | 0.274 | 0.434 | 0.414 | 4.387 | 4.724 |
| 4 | 0.066 | 0.171 | 0.251 | 0.421 | 0.402 | 4.363 | 4.690 |
| 8 | 0.061 | 0.180 | 0.313 | 0.434 | 0.414 | 4.509 | 4.872 |
| 16 | 0.059 | 0.178 | 0.280 | 0.437 | 0.417 | 4.415 | 4.849 |

准备相关阶段中，`batch_prepare + input_pack` 每步约 0.22 至 0.24 ms，只占约
4.3 至 4.5 ms 的 `step_total` 的 5% 左右；`update_overlap_prepare` 在五档间也只有
约 0.03 ms 的变化。相比之下，`embed_lookup` 和 `sparse_optimizer` 的轮间波动更大，
且它们不受 lookahead 队列深度直接控制。这与“planned lookup 已 resident、lookahead
只影响未来 batch 准备队列”的实现语义一致：把队列从 0 加深到 15 个位置，并未缩短
主验收口径下的关键路径。

如果目标是默认参数选择，la4 仍是稳妥选择。若更关心 host 侧预取队列占用，la1 在本轮
没有表现出明显吞吐损失，但 +0.89% 的主口径差异不足以证明它更快；建议后续用 la1 与
la4 交错运行、增加重复次数后再确认。当前 3 次顺序重复足以排除大幅收益或大幅回退，
不足以量化小于约 2% 的差异。
