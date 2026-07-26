# HSTU ML-1M RecStore smoke 运行记录

- 日期：2026-07-24（Asia/Shanghai）
- 容器：`liuxiaoyu-hstu-recstore`
- GPU 映射：宿主机 GPU 1（`GPU-4e3a8414-ec19-c000-ed69-94ee644c4747`）→ 容器 GPU 0
- 启动前显存：已用 4 MiB，空闲 24079 MiB
- 范围：3 epoch smoke；不运行 101 epoch 完整训练
- PS：GRPC 两 shard，`127.0.0.1:15120` 和 `127.0.0.1:15121`
- PS 配置：`recstore_hstu_ml1m_v1_config.json`
- Gin 配置：`configs/ml-1m/hstu-sampled-softmax-n128-recstore-smoke.gin`
- 训练日志：容器内 `/workspace/run/hstu-ml1m-smoke/train.log`

备注：容器已正确绑定到当前空闲的宿主机 GPU 1，因此未重建或改动其他容器。

## 结果

- 状态：通过，进程退出码 0，无 `Traceback`、`ERROR` 或 OOM
- 最终训练：`losses/ar_loss=3.99056411`，step 143
- 最终评估（epoch step 2）：
  - `HR@10=0.03807947`，`NDCG@10=0.01929775`
  - `HR@50=0.11837748`，`NDCG@50=0.03617622`
  - `HR@200=0.31854305`，`NDCG@200=0.06596301`
- 清理：smoke PS 已停止；原 `15100/15101` PS 未改动；未启动 101 epoch 完整训练
