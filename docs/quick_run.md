# 快速运行

本指南用于在完成[搭建指南](quickstart.md)后，进行全流程的运行，相关步骤出现错误可以查看 [FAQ](./faq.md)。

## 1. 启动参数服务器

在仓库根目录执行：

```bash title="自动读取配置文件，启动ps客户端"
./buidl/bin/ps_server
```

## 2. 运行计算层模型

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
