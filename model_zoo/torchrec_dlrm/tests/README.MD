# RecStore Test Cases List

## RecStoreEmbeddingBagCollection CPU Test Cases

<!-- > based on Commit 7000e63b71c7071e1a38848cd8cb4f351739ee9e -->
> based on Commit 9c31e49a4e692ea47caa6452f43124fc3cf6e678

| Test Case | Test Objective | Status | Description |
|-----------|---------------|--------|-------------|
| test_basic_initialization | Basic initialization functionality | ✅ | Verify single table configuration initialization |
| test_multi_table_initialization | Multi-table initialization | ✅ | Verify multi-table configuration initialization |
<!-- | test_empty_config_error | Empty configuration error handling | ✅ | Verify exception thrown for empty configuration | -->
<!-- | test_missing_config_fields | Missing configuration fields error handling | ✅ | Verify exception thrown for missing required fields | -->
| test_basic_forward_pass | Basic forward pass | ✅ | Verify basic embedding lookup |
| test_multi_table_forward_pass | Multi-table forward pass | ✅ | Verify multi-table embedding lookup |
| test_empty_batch_forward_pass | Empty batch forward pass | ✅ | Verify empty batch processing |
| test_single_id_forward_pass | Single ID forward pass | ✅ | Verify single ID processing |
| test_gradient_update | Gradient update functionality | ✅ | Verify backpropagation and gradient update |
| test_multi_table_gradient_update | Multi-table gradient update | ✅ | Verify multi-table gradient update |
| test_direct_kv_client_operations | Direct KV client operations | ✅ | Verify pull/push/update operations |
| test_embedding_consistency | Embedding consistency | ✅ | Verify embedding bag averaging operation |
| test_large_batch_forward_pass | Large batch forward pass | ✅ | Verify large batch processing |
| test_repr_function | repr function | ✅ | Verify object string representation |
| test_device_handling | Device handling | ✅ | Verify CPU device compatibility |
| test_dtype_handling | Data type handling | ✅ | Verify data type conversion |
| test_error_handling | Error handling | ✅ | Verify exception handling |
| test_zero_length_forward_pass | Zero length forward pass | ✅ | Verify zero length input processing |
| test_multiple_ids_per_sample | Multiple IDs per sample | ✅ | Verify multi-ID sample processing |
| test_embedding_update_consistency | Embedding update consistency | ✅ | Verify post-update consistency |

## RecStoreEmbeddingBagCollection GPU Test Cases

> based on Commit 0aafa8115ff2ca1200eceaedf632a630286194d0

| Test Case | Test Objective | Status | Description |
|-----------|---------------|--------|-------------|
| test_gpu_initialization | GPU initialization functionality | ✅ | Verify single table configuration initialization on GPU |
| test_gpu_forward_pass | GPU forward pass | ✅ | Verify basic embedding lookup on GPU |
| test_gpu_multi_table_forward_pass | GPU multi-table forward pass | ✅ | Verify multi-table embedding lookup on GPU |
| test_gpu_gradient_update | GPU gradient update functionality | ✅ | Verify backpropagation and gradient update on GPU |
| test_gpu_multi_table_gradient_update | GPU multi-table gradient update | ✅ | Verify multi-table gradient update on GPU |
| test_gpu_device_transfer | GPU device data transfer | ✅ | Verify CPU to GPU data transfer |
| test_gpu_large_batch_forward_pass | GPU large batch forward pass | ✅ | Verify large batch processing on GPU |
| test_gpu_memory_management | GPU memory management | ✅ | Verify GPU memory allocation and deallocation |
| test_gpu_mixed_precision | GPU mixed precision | ✅ | Verify mixed precision computation on GPU |
| test_gpu_concurrent_operations | GPU concurrent operations | ✅ | Verify multi-stream concurrent processing on GPU |
| test_gpu_error_handling | GPU error handling | ✅ | Verify exception handling on GPU |
| test_gpu_performance_benchmark | GPU performance benchmark | ✅ | Verify GPU performance testing |
| test_gpu_multi_device | Multi-GPU device | ✅ | Verify multi-GPU device support |
| test_gpu_tensor_operations | GPU tensor operations | ✅ | Verify tensor operations on GPU |
| test_gpu_gradient_accumulation | GPU gradient accumulation | ✅ | Verify gradient accumulation on GPU |

## TorchRec Official Version Comparison Tests

> based on Commit cdb0ccddeae66cf2ae93410a1f26ce125c856413

### TorchRec vs RecStore Behavior Comparison

| Test Scenario | TorchRec Official Version | RecStore Version | Difference Description |
|---------------|---------------------------|------------------|----------------------|
| **Basic Forward Pass** | | | |
| Output Shape | `(batch_size, embedding_dim)` | `(batch_size, embedding_dim)` | Same |
| length_per_key | `[embedding_dim]` | `[batch_size]` | **Different**: TorchRec represents embedding dimension, RecStore represents sample count |
| **Multi-table Forward Pass** | | | |
| Output Shape | `(batch_size, total_embedding_dim)` | `(batch_size * num_features, embedding_dim)` | **Different**: TorchRec concatenates all features, RecStore processes separately |
| length_per_key | `[dim1, dim2, dim3]` | `[batch_size, batch_size, batch_size]` | **Different**: TorchRec represents feature dimensions, RecStore represents feature sample counts |
| **Empty Configuration Handling** | | | |
| Empty Config | May not throw exception | Throws `ValueError` | **Different**: Different error handling strategies |
| **Out-of-range IDs** | | | |
| Invalid ID (999) | Throws `RuntimeError` | Returns zero vectors | **Different**: TorchRec strict checking, RecStore fault-tolerant handling |
| **KV Client Operations** | | | |
| Direct KV Operations | Not supported | Supports pull/push/update | **Different**: RecStore provides additional KV interface |
| **Embedding Updates** | | | |
| Embedding Consistency | Standard PyTorch embeddings | Custom KV storage | **Different**: Different storage backends |
| **Large Batch Processing** | | | |
| Large Batch Shape | `(batch_size, embedding_dim)` | `(batch_size, embedding_dim)` | Same |
| **Device Handling** | | | |
| CPU/GPU Support | Standard PyTorch devices | Standard PyTorch devices | Same |
| **Data Types** | | | |
| Input/Output Types | Standard PyTorch types | Standard PyTorch types | Same |

### Test Statistics

### CPU Test Statistics
- **Total Tests**: 18
- **Passed Tests**: 18
- **Failed Tests**: 0
- **Pass Rate**: 100%

### GPU Test Statistics
- **Total Tests**: 15
- **Passed Tests**: 15
- **Failed Tests**: 0
- **Pass Rate**: 100%

## Known Limitations

- All tables must use the same embedding dimension (KVClientOp singleton limitation)
- Distributed KV storage not supported
- RecStore version has behavioral differences from TorchRec official version in some interfaces, but core functionality remains consistent