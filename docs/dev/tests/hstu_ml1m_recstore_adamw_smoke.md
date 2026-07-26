# HSTU ML-1M RecStore AdamW smoke

- 日期：2026-07-25（Asia/Shanghai）
- 容器：`liuxiaoyu-hstu-recstore`
- GPU 映射：宿主机 GPU 1（`GPU-4e3a8414-ec19-c000-ed69-94ee644c4747`）→ 容器 GPU 0
- 范围：3 epoch smoke；未运行 101 epoch 完整训练
- AdamW PS：专用 gRPC shard `127.0.0.1:15140`、`127.0.0.1:15141`
- 训练配置：`recstore_hstu_ml1m_adamw_smoke_config.json` +
  `configs/ml-1m/hstu-sampled-softmax-n128-recstore-adamw-smoke.gin`
- 训练 event：
  `exps/ml-1m-l200/HSTU-b8-h2-dqk25-dv25-lsilud0.2-ad0.0-recstore_DotProduct_local-l2-eps1e-06_ssl-t0.05-n128-b128-lr0.001-wu0-wd0-2026-07-25-hstu_ml1m_items_adamw_smoke/`

## 结果

RecStore sparse AdamW（`lr=1e-3`，`beta1=0.9`，`beta2=0.98`，`eps=1e-8`，
`weight_decay=0`）在 3 epoch 结束时：

- `losses/ar_loss=3.09749413`，step 143
- `HR@10=0.15281457`，`NDCG@10=0.08254541`
- `HR@50=0.36440396`，`NDCG@50=0.12834527`
- `HR@200=0.60894042`，`NDCG@200=0.16529034`

同一官方 local AdamW event 的 3 epoch 结果为：

- `losses/ar_loss=3.08697319`
- `HR@10=0.15447021`，`NDCG@10=0.08102281`
- `HR@50=0.35099337`，`NDCG@50=0.12383651`
- `HR@200=0.60430461`，`NDCG@200=0.16189297`

相较此前 RecStore RowWiseAdagrad smoke 的 `loss=3.99056411`，AdamW 已显著接近官方 AdamW。
两者不会逐 step 完全相同：当前 RecStore AdamW 按稀疏语义只推进本 batch 命中的行；官方
`torch.optim.AdamW` 对 dense embedding 参数张量的每一步都推进所有行的 moment/decay。
因此这次 smoke 验证的是优化方向和指标改善，不宣称已经完成 dense AdamW 的数值对齐。

## 验证

- C++ `bin/test_optimizer`：3/3 passed，包含重建 optimizer 后从持久 step 继续的检查。
- Python `test_sparse_optimizer_config`：3/3 passed。
- smoke 无 `Traceback`、`ERROR` 或 OOM。
- AdamW smoke PS 已停止；原有 15100/15101 PS 保持运行。
