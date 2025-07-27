# RecStore 测试用例清单

## RecStoreEmbeddingBagCollection CPU测试用例

> based on Commit 7000e63b71c7071e1a38848cd8cb4f351739ee9e

| 测试用例 | 测试目标 | 状态 | 说明 |
|---------|---------|------|------|
| test_basic_initialization | 基本初始化功能 | ✅ | 验证单表配置初始化 |
| test_multi_table_initialization | 多表初始化 | ✅ | 验证多表配置初始化 |
| test_empty_config_error | 空配置错误处理 | ✅ | 验证空配置抛出异常 |
| test_missing_config_fields | 缺少配置字段错误处理 | ✅ | 验证缺少必要字段抛出异常 |
| test_basic_forward_pass | 基本前向传播 | ✅ | 验证基本embedding lookup |
| test_multi_table_forward_pass | 多表前向传播 | ✅ | 验证多表embedding lookup |
| test_empty_batch_forward_pass | 空批次前向传播 | ✅ | 验证空批次处理 |
| test_single_id_forward_pass | 单个ID前向传播 | ✅ | 验证单个ID处理 |
| test_gradient_update | 梯度更新功能 | ✅ | 验证反向传播和梯度更新 |
| test_multi_table_gradient_update | 多表梯度更新 | ✅ | 验证多表梯度更新 |
| test_direct_kv_client_operations | 直接KV客户端操作 | ✅ | 验证pull/push/update操作 |
| test_embedding_consistency | 嵌入一致性 | ✅ | 验证embedding bag平均操作 |
| test_large_batch_forward_pass | 大批次前向传播 | ✅ | 验证大批次处理 |
| test_repr_function | repr函数 | ✅ | 验证对象字符串表示 |
| test_device_handling | 设备处理 | ✅ | 验证CPU设备兼容性 |
| test_dtype_handling | 数据类型处理 | ✅ | 验证数据类型转换 |
| test_error_handling | 错误处理 | ✅ | 验证异常情况处理 |
| test_zero_length_forward_pass | 零长度前向传播 | ✅ | 验证零长度输入处理 |
| test_multiple_ids_per_sample | 每个样本多个ID | ✅ | 验证多ID样本处理 |
| test_embedding_update_consistency | 嵌入更新一致性 | ✅ | 验证更新后一致性 |

## RecStoreEmbeddingBagCollection GPU测试用例

> based on Commit 0aafa8115ff2ca1200eceaedf632a630286194d0

| 测试用例 | 测试目标 | 状态 | 说明 |
|---------|---------|------|------|
| test_gpu_initialization | GPU初始化功能 | ✅ | 验证GPU上单表配置初始化 |
| test_gpu_forward_pass | GPU前向传播 | ✅ | 验证GPU上基本embedding lookup |
| test_gpu_multi_table_forward_pass | GPU多表前向传播 | ✅ | 验证GPU上多表embedding lookup |
| test_gpu_gradient_update | GPU梯度更新功能 | ✅ | 验证GPU上反向传播和梯度更新 |
| test_gpu_multi_table_gradient_update | GPU多表梯度更新 | ✅ | 验证GPU上多表梯度更新 |
| test_gpu_device_transfer | GPU设备间数据传输 | ✅ | 验证CPU到GPU数据传输 |
| test_gpu_large_batch_forward_pass | GPU大批次前向传播 | ✅ | 验证GPU上大批次处理 |
| test_gpu_memory_management | GPU内存管理 | ✅ | 验证GPU内存分配和释放 |
| test_gpu_mixed_precision | GPU混合精度 | ✅ | 验证GPU上混合精度计算 |
| test_gpu_concurrent_operations | GPU并发操作 | ✅ | 验证GPU上多流并发处理 |
| test_gpu_error_handling | GPU错误处理 | ✅ | 验证GPU上异常情况处理 |
| test_gpu_performance_benchmark | GPU性能基准 | ✅ | 验证GPU性能测试 |
| test_gpu_multi_device | 多GPU设备 | ✅ | 验证多GPU设备支持 |
| test_gpu_tensor_operations | GPU张量操作 | ✅ | 验证GPU上张量运算 |
| test_gpu_gradient_accumulation | GPU梯度累积 | ✅ | 验证GPU上梯度累积 |

## 测试统计

### CPU测试统计
- **总测试数**: 20
- **通过测试**: 20
- **失败测试**: 0
- **通过率**: 100%

### GPU测试统计
- **总测试数**: 15
- **通过测试**: 15
- **失败测试**: 0
- **通过率**: 100%

## 已知限制

- 所有表必须使用相同的嵌入维度（KVClientOp单例限制）
- 不支持分布式KV存储