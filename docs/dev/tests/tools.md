# 测试用工具

## ps_server_helpers.py

`src/test/scripts/ps_server_helpers.py` 模块提供了一系列辅助函数，用于在测试环境中查找、检测和配置 `ps_server`。端口探活与启动决策实际委托给 C++ 侧启动器 `ps_server_launcher_cli`（见下文），Python 侧负责读取配置、整理 RDMA runner 参数并在 CI 中判定跳过。

| 变量名 | 说明 | 默认值 |
| :--- | :--- | :--- |
| `RECSTORE_CONFIG` | 指定 `recstore_config.json` 路径 | 自动查找 |
| `PS_LOG_DIR` | 指定日志输出目录 | `/tmp/recstore_ps` |
| `PS_TIMEOUT` | 启动超时时间（秒） | 60 |
| `PS_NUM_SHARDS` | 期望的分片数量 | 2 |

### `find_ps_server_binary()` / `find_ps_server_launcher_cli()`

返回 `build/bin/` 下的 `ps_server` 与 `ps_server_launcher_cli` 可执行文件路径。

### `run_launcher_decision(config_path=None)`

调用 `ps_server_launcher_cli decision` 并解析其 JSON 输出，返回启动决策（是否启动、失败原因、配置端口、已开放端口等）。返回码 `2` 表示“判定失败”，会在结果里带上 `should_fail`。

### `find_config_file()` / `load_config()`

按 `RECSTORE_CONFIG` → 自当前目录向上查找 `recstore_config.json` → 仓库根目录的顺序定位当前配置；`load_config()` 返回 `(path, dict)`，找不到时返回 `(None, {})`。

### `get_backend_type()`

返回 `cache_ps.ps_type`（大写），默认 `GRPC`。

### `get_rdma_runner_config()`

从配置中提取 PetPS RDMA 测试所需的参数，并对 `distributed_client` / `rdma_deployment` 做严格校验，缺失或非法时直接抛 `ValueError`。返回字段：

| 键名 | 来源 |
| :--- | :--- |
| `num_servers` | `distributed_client.num_shards` |
| `value_size` | `base_kv_config.value.default_value_size_hint`（回退 `value_size` / 512） |
| `max_kv_num_per_request` | `distributed_client.max_keys_per_request` |
| `num_clients` | `rdma_deployment.num_clients` |
| `rdma_deployment_nodes` | `rdma_deployment.nodes` |

校验项包括：`num_shards` / `max_keys_per_request` / `num_clients` / `epoch` 为正整数，`rdma_deployment.nodes` 非空，`deployment_id` 非空，`protocol_version == 1`。

### `get_rdma_skip_reason()`

当 `/dev/infiniband` 下没有 `uverbs*` 设备时返回跳过原因，否则返回 `None`；配套常量 `RDMA_SKIP_EXIT_CODE = 77`。

### `get_ports_from_config()` / `check_ps_server_running(ports=None)`

基于启动器决策读取配置端口并探活。`check_ps_server_running` 返回 `(is_running: bool, open_ports: list)`。

### `should_skip_server_start()`

根据启动器决策判断是否跳过启动 `ps_server`，返回 `(skip: bool, reason: str)`。判定失败时抛 `RuntimeError`；`already_running` / `ci_reuse_running` / `NO_PS_SERVER` 直接跳过；CI 下 `ci_server_not_ready` 会以明确原因返回不跳过。

### `get_server_config()`

返回标准化服务器配置字典：

| 键名 | 说明 |
| :--- | :--- |
| `server_path` | `ps_server` 二进制路径 |
| `launcher_cli` | `ps_server_launcher_cli` 路径 |
| `config_path` | 配置文件路径（`RECSTORE_CONFIG`） |
| `log_dir` | 日志目录（`PS_LOG_DIR`） |
| `timeout` | 超时时间（`PS_TIMEOUT`） |
| `num_shards` | 分片数量（`PS_NUM_SHARDS`） |

## ps_server_launcher (C++)

`src/test/server_mgr/ps_server_launcher.h` 与 `src/test/server_mgr/ps_server_launcher.cpp` 提供了 C++ 侧的 `ps_server` 启停能力，适合 C++ 单测与工具程序复用。

该模块与 Python 侧环境变量语义保持一致（便于混合测试场景复用同一套环境配置）。

| 变量名 | 说明 | 默认值 |
| :--- | :--- | :--- |
| `PS_SERVER_PATH` | 指定 `ps_server` 的绝对路径 | 自动查找 |
| `RECSTORE_CONFIG` | 指定 `recstore_config.json` 路径 | 自动查找 |
| `PS_LOG_DIR` | 指定日志输出目录 | `./logs` |
| `PS_TIMEOUT` | 启动超时时间（秒） | 60 |
| `PS_NUM_SHARDS` | 期望的分片数量 | 2 |
| `PS_SERVER_PS_TYPE` | 启动前临时覆盖 `cache_ps.ps_type`（如 `GRPC`/`BRPC`） | 不覆盖 |
| `PS_SERVER_PORTS` | 启动前临时覆盖服务端口，逗号分隔（如 `15123,15124`） | 不覆盖 |
| `NO_PS_SERVER` | 强制跳过启动服务器 | False |

### 主要能力

- 端口探活与启动决策（支持“部分端口打开”直接判错）。
- 基于日志的 shard ready 检测。
- 进程生命周期管理（优雅停止 + 超时强杀）。
- RAII 封装（`ScopedPSServer`）。

### 关键类型

| 类型 | 说明 |
| :--- | :--- |
| `LauncherOptions` | 启动参数（路径、超时、分片数等） |
| `LaunchDecision` | 启动决策结果（是否启动、失败原因、端口状态） |
| `LaunchResult` | 启动结果（PID、ready 分片、日志路径） |
| `PSServerLauncher` | 启停与状态查询主类 |
| `ScopedPSServer` | RAII 封装，作用域结束自动停止 |

### CMake 目标

- 库目标: `ps_server_mgr`
- 测试目标: `test_ps_server_launcher`

### 最小用法

```cpp
#include "ps_server_launcher.h"

using namespace recstore::test;

void RunTestCase() {
  LauncherOptions opts = PSServerLauncher::LoadOptionsFromEnvironment();
  ScopedPSServer server(opts, true);

  // test logic here
}
```

### 协议配置建议

- BRPC 客户端测试默认可使用仓库根目录的 `recstore_config.json`。
- gRPC 客户端测试建议在启动器里设置 `LauncherOptions.override_ps_type = "GRPC"`，必要时配合 `LauncherOptions.override_ports`（或 `PS_SERVER_PORTS`）使用独立端口，避免复用已占用端口导致协议不匹配。

### 目录说明

测试模块目录已统一为 `src/test/server_mgr`（原先较长命名 `server_management` 已替换）。

## analyze_embupdate_stages.py

`src/test/scripts/analyze_embupdate_stages.py` 用于分析 update 链路的分阶段性能数据。  
它会从本地 `REPORT_LOCAL_EVENT` 日志或 JSONL 事件文件中提取 `embupdate_stages` 表数据，并输出:

- 各阶段指标统计（mean/p50/p95/p99/max）
- 近似拆分（序列化、后端执行、网络/框架开销）
- 慢请求 TopN（按 trace 维度）

### 输入来源

支持两种输入格式:

1. glog 文本日志（包含 `REPORT_LOCAL_EVENT {...}` 行）
2. 纯 JSONL（每行一个事件 JSON）

### 常用命令

```bash
python3 src/test/scripts/analyze_embupdate_stages.py --input /path/to/recstore.log
```

```bash
python3 src/test/scripts/analyze_embupdate_stages.py \
  --input /path/to/server.log \
  --input /path/to/client.log \
  --top 20
```

```bash
python3 src/test/scripts/analyze_embupdate_stages.py \
  --input /path/to/report_events.jsonl \
  --trace-prefix grpc_client::EmbUpdate
```

```bash
python3 src/test/scripts/analyze_embupdate_stages.py \
  --input /path/to/report_events.jsonl \
  --group-by-prefix \
  --export-csv /tmp/embupdate_report.csv
```

### 关键指标解读

- `client_serialize_us`: 客户端将梯度打包成请求的耗时。
- `client_rpc_us`: 客户端从发起 RPC 到返回的耗时（包含网络与服务端处理）。
- `server_total_us`: 服务端 `UpdateParameter` 总耗时。
- `server_backend_update_us`: 服务端后端更新逻辑（cache/storage/update）执行耗时。
- `op_total_us`: op 层（`EmbUpdate`）总耗时。

近似网络/框架开销:

`client_rpc_us - server_total_us`

这有助于快速判断瓶颈更偏网络侧还是服务端执行侧。
