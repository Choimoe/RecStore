# 本次实验分析

主验收吞吐 = 总样本 / max_rank(sum(step_wall_ms))，包含 barrier 和每步收尾。CSV stage 耗时吞吐不含这些尾部开销，只供辅助分析。事件窗口吞吐包含步间空隙；三者均不含初始化与关闭。Profiler 运行只用于归因。

| Run | Backend | Profile | Steps | Endpoint wall samples/s | CSV stage samples/s | Event wall samples/s | p95 slow step ms |
|---|---|---|---:|---:|---:|---:|---:|
| /home/hadoop-quanta/VSCodeProjects/RecStore_shanhongqi/results/lookahead_e2e_10251621/la1/outputs/rdma_b1024_d128_r0 | recstore | False | 300 | 443314.3 | 457839.7 | unavailable | 4.706 |
| /home/hadoop-quanta/VSCodeProjects/RecStore_shanhongqi/results/lookahead_e2e_10251621/la1/outputs/rdma_b1024_d128_r1 | recstore | False | 300 | 449295.1 | 465353.7 | unavailable | 4.664 |
| /home/hadoop-quanta/VSCodeProjects/RecStore_shanhongqi/results/lookahead_e2e_10251621/la1/outputs/rdma_b1024_d128_r2 | recstore | False | 300 | 436630.5 | 452895.7 | unavailable | 4.811 |
| /home/hadoop-quanta/VSCodeProjects/RecStore_shanhongqi/results/lookahead_e2e_10251621/la2/outputs/rdma_b1024_d128_r0 | recstore | False | 300 | 438677.7 | 452898.2 | unavailable | 4.957 |
| /home/hadoop-quanta/VSCodeProjects/RecStore_shanhongqi/results/lookahead_e2e_10251621/la2/outputs/rdma_b1024_d128_r1 | recstore | False | 300 | 409640.8 | 431364.5 | unavailable | 4.956 |
| /home/hadoop-quanta/VSCodeProjects/RecStore_shanhongqi/results/lookahead_e2e_10251621/la2/outputs/rdma_b1024_d128_r2 | recstore | False | 300 | 449218.3 | 463528.4 | unavailable | 4.595 |
| /home/hadoop-quanta/VSCodeProjects/RecStore_shanhongqi/results/lookahead_e2e_10251621/la4/outputs/rdma_b1024_d128_r0 | recstore | False | 300 | 451211.5 | 465637.8 | unavailable | 4.680 |
| /home/hadoop-quanta/VSCodeProjects/RecStore_shanhongqi/results/lookahead_e2e_10251621/la4/outputs/rdma_b1024_d128_r1 | recstore | False | 300 | 438264.3 | 457932.4 | unavailable | 4.700 |
| /home/hadoop-quanta/VSCodeProjects/RecStore_shanhongqi/results/lookahead_e2e_10251621/la4/outputs/rdma_b1024_d128_r2 | recstore | False | 300 | 428052.3 | 441431.7 | unavailable | 4.900 |
| /home/hadoop-quanta/VSCodeProjects/RecStore_shanhongqi/results/lookahead_e2e_10251621/la8/outputs/rdma_b1024_d128_r0 | recstore | False | 300 | 448281.4 | 462912.8 | unavailable | 4.729 |
| /home/hadoop-quanta/VSCodeProjects/RecStore_shanhongqi/results/lookahead_e2e_10251621/la8/outputs/rdma_b1024_d128_r1 | recstore | False | 300 | 408625.8 | 420789.5 | unavailable | 5.178 |
| /home/hadoop-quanta/VSCodeProjects/RecStore_shanhongqi/results/lookahead_e2e_10251621/la8/outputs/rdma_b1024_d128_r2 | recstore | False | 300 | 422916.4 | 436231.0 | unavailable | 4.950 |
| /home/hadoop-quanta/VSCodeProjects/RecStore_shanhongqi/results/lookahead_e2e_10251621/la16/outputs/rdma_b1024_d128_r0 | recstore | False | 300 | 433191.5 | 448800.9 | unavailable | 4.901 |
| /home/hadoop-quanta/VSCodeProjects/RecStore_shanhongqi/results/lookahead_e2e_10251621/la16/outputs/rdma_b1024_d128_r1 | recstore | False | 300 | 430781.7 | 446080.4 | unavailable | 5.025 |
| /home/hadoop-quanta/VSCodeProjects/RecStore_shanhongqi/results/lookahead_e2e_10251621/la16/outputs/rdma_b1024_d128_r2 | recstore | False | 300 | 435347.5 | 449594.3 | unavailable | 4.881 |

完整 rank × step 校验错误：
