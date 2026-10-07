# 快速运行

本指南用于在完成[搭建指南](quickstart.md)后，进行全流程的运行，相关步骤出现错误可以查看 [常见问题](./faq.md)。

## 0. 环境准备

请确认全局 PyTorch 环境已经安装：

```bash
python3 -c "import torch, torchrec, fbgemm_gpu, torchmetrics; print(torch.__version__, torch.version.cuda, torch.compiled_with_cxx11_abi())"
```

推荐输出：

```text
2.7.1+cu118 11.8 True
```

## 1. 启动参数服务器

在仓库根目录执行：

```bash title="自动读取配置文件，启动ps客户端"
./build/bin/ps_server
```

配置文件相关信息可以参考：[项目配置](config.md)

## 2. 运行计算层模型

DLRM 使用的数据集为 [Criteo Kaggle Display Advertising Challenge Dataset](https://ailab.criteo.com/ressources/)，项目切片了第 0 天的数据方便进行测试和分析，你可以在 [day_0.csv](https://github.com/user-attachments/files/23355355/day_0.csv) 下载前 4096 条的数据，然后使用脚本进行预处理：

```bash title="预处理./partial_data中的数据"
cd model_zoo/torchrec_dlrm/
mkdir -p partial_data && wget -O partial_data/day_0 https://github.com/user-attachments/files/23355355/day_0.csv
bash scripts/process_single_day.sh ./partial_data ./processed_day_0_data > process.log 2>&1
```

来完成数据集的加载，随后可以直接进行训练。

=== "RecStore"
    ```bash
    cd model_zoo/torchrec_dlrm/
    bash run_single_day.sh
    ```
=== "TorchRec"
    ```bash
    cd model_zoo/torchrec_dlrm/
    bash run_single_day.sh --torchrec
    ```

可以使用 `--help` 参数来获取支持的所有参数。

## 3. 下一步

你可以：

- 在 [性能分析](dev/performance.md) 阅读更详细的性能分析工具。
- 在 [分布式训练](dev/distributed.md) 查看如何使用分布式 RecStore。
