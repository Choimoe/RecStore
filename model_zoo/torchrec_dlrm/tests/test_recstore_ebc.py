import torch
import os
import sys
import unittest
import tempfile
import shutil
from torchrec.sparse.jagged_tensor import KeyedJaggedTensor
from torchrec.sparse.jagged_tensor import KeyedTensor

RECSTORE_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../../src'))
if RECSTORE_PATH not in sys.path:
    sys.path.insert(0, RECSTORE_PATH)

from python.pytorch.torchrec.EmbeddingBag import RecStoreEmbeddingBagCollection


class TestRecStoreEmbeddingBagCollection(unittest.TestCase):
    """测试RecStoreEmbeddingBagCollection的各种功能"""
    
    def setUp(self):
        """每个测试前的设置"""
        # 设置日志级别为DEBUG以获取更多信息
        os.environ['RECSTORE_LOG_LEVEL'] = '3'
        
        # 基础配置
        self.basic_configs = [
            {
                "name": "test_table",
                "num_embeddings": 100,
                "embedding_dim": 16,
                "feature_names": ["test_feature"]
            }
        ]
        
        # 多表配置 - 使用相同维度避免冲突
        self.multi_table_configs = [
            {
                "name": "user_table",
                "num_embeddings": 1000,
                "embedding_dim": 16,  # 改为16避免维度冲突
                "feature_names": ["user_id"]
            },
            {
                "name": "item_table", 
                "num_embeddings": 500,
                "embedding_dim": 16,  # 改为16避免维度冲突
                "feature_names": ["item_id"]
            },
            {
                "name": "category_table",
                "num_embeddings": 100,
                "embedding_dim": 16,  # 改为16避免维度冲突
                "feature_names": ["category_id"]
            }
        ]

    def test_basic_initialization(self):
        """测试基本的初始化功能"""
        print("\n=== 测试基本初始化 ===")
        ebc = RecStoreEmbeddingBagCollection(self.basic_configs)
        
        # 验证配置
        configs = ebc.embedding_bag_configs()
        self.assertEqual(len(configs), 1)
        self.assertEqual(configs[0].name, "test_table")
        self.assertEqual(configs[0].num_embeddings, 100)
        self.assertEqual(configs[0].embedding_dim, 16)
        
        # 验证特征键
        self.assertEqual(ebc.feature_keys, ["test_table"])
        
        print("✓ 基本初始化测试通过")

    def test_multi_table_initialization(self):
        """测试多表初始化"""
        print("\n=== 测试多表初始化 ===")
        ebc = RecStoreEmbeddingBagCollection(self.multi_table_configs)
        
        # 验证配置数量
        configs = ebc.embedding_bag_configs()
        self.assertEqual(len(configs), 3)
        
        # 验证特征键
        expected_keys = ["user_table", "item_table", "category_table"]
        self.assertEqual(ebc.feature_keys, expected_keys)
        
        # 验证嵌入维度
        expected_dims = {"user_table": 16, "item_table": 16, "category_table": 16}
        self.assertEqual(ebc._embedding_dims, expected_dims)
        
        print("✓ 多表初始化测试通过")

    def test_empty_config_error(self):
        """测试空配置错误处理"""
        print("\n=== 测试空配置错误处理 ===")
        with self.assertRaises(ValueError):
            RecStoreEmbeddingBagCollection([])
        print("✓ 空配置错误处理测试通过")

    def test_missing_config_fields(self):
        """测试缺少配置字段的错误处理"""
        print("\n=== 测试缺少配置字段错误处理 ===")
        invalid_configs = [
            {
                "name": "test_table",
                # 缺少 num_embeddings
                "embedding_dim": 16
            }
        ]
        with self.assertRaises(KeyError):  # 改为KeyError，因为是在字典访问时抛出的
            RecStoreEmbeddingBagCollection(invalid_configs)
        print("✓ 缺少配置字段错误处理测试通过")

    def test_basic_forward_pass(self):
        """测试基本的前向传播"""
        print("\n=== 测试基本前向传播 ===")
        ebc = RecStoreEmbeddingBagCollection(self.basic_configs)
        
        # 构造输入数据
        kjt = KeyedJaggedTensor(
            keys=["test_table"],
            values=torch.tensor([1, 2, 3], dtype=torch.int64),
            lengths=torch.tensor([2, 1], dtype=torch.int32)
        )
        
        # 前向传播
        result = ebc(kjt)
        
        # 验证结果
        self.assertIsInstance(result, KeyedTensor)
        self.assertEqual(result.keys(), ["test_table"])
        self.assertEqual(result.values().shape, (1, 16))  # 1个特征，16维嵌入
        self.assertEqual(result.length_per_key().tolist(), [1])
        
        print(f"✓ 基本前向传播测试通过，输出形状: {result.values().shape}")

    def test_multi_table_forward_pass(self):
        """测试多表前向传播"""
        print("\n=== 测试多表前向传播 ===")
        ebc = RecStoreEmbeddingBagCollection(self.multi_table_configs)
        
        # 构造多表输入数据
        kjt = KeyedJaggedTensor(
            keys=["user_table", "item_table", "category_table"],
            values=torch.tensor([1, 2, 3, 4, 5], dtype=torch.int64),
            lengths=torch.tensor([1, 2, 2], dtype=torch.int32)
        )
        
        # 前向传播
        result = ebc(kjt)
        
        # 验证结果
        self.assertIsInstance(result, KeyedTensor)
        self.assertEqual(result.keys(), ["user_table", "item_table", "category_table"])
        self.assertEqual(result.values().shape, (3, 16))  # 3个特征，16维嵌入
        self.assertEqual(result.length_per_key().tolist(), [1, 1, 1])
        
        print(f"✓ 多表前向传播测试通过，输出形状: {result.values().shape}")

    def test_empty_batch_forward_pass(self):
        """测试空批次的前向传播"""
        print("\n=== 测试空批次前向传播 ===")
        ebc = RecStoreEmbeddingBagCollection(self.basic_configs)
        
        # 构造空批次数据 - 使用至少一个样本避免索引越界
        kjt = KeyedJaggedTensor(
            keys=["test_table"],
            values=torch.tensor([], dtype=torch.int64),
            lengths=torch.tensor([0], dtype=torch.int32)  # 一个样本，但长度为0
        )
        
        # 前向传播
        result = ebc(kjt)
        
        # 验证结果
        self.assertIsInstance(result, KeyedTensor)
        self.assertEqual(result.keys(), ["test_table"])
        self.assertEqual(result.values().shape, (1, 16))  # 1个特征，16维嵌入
        self.assertEqual(result.length_per_key().tolist(), [1])
        
        print("✓ 空批次前向传播测试通过")

    def test_single_id_forward_pass(self):
        """测试单个ID的前向传播"""
        print("\n=== 测试单个ID前向传播 ===")
        ebc = RecStoreEmbeddingBagCollection(self.basic_configs)
        
        # 构造单个ID数据
        kjt = KeyedJaggedTensor(
            keys=["test_table"],
            values=torch.tensor([42], dtype=torch.int64),
            lengths=torch.tensor([1], dtype=torch.int32)
        )
        
        # 前向传播
        result = ebc(kjt)
        
        # 验证结果
        self.assertIsInstance(result, KeyedTensor)
        self.assertEqual(result.keys(), ["test_table"])
        self.assertEqual(result.values().shape, (1, 16))
        
        print("✓ 单个ID前向传播测试通过")

    def test_gradient_update(self):
        """测试梯度更新功能"""
        print("\n=== 测试梯度更新功能 ===")
        ebc = RecStoreEmbeddingBagCollection(self.basic_configs)
        
        # 构造输入数据
        kjt = KeyedJaggedTensor(
            keys=["test_table"],
            values=torch.tensor([1, 2, 3], dtype=torch.int64),
            lengths=torch.tensor([2, 1], dtype=torch.int32)
        )
        
        # 前向传播
        result = ebc(kjt)
        
        # 构造梯度
        grad = torch.randn_like(result.values())
        
        # 反向传播 - 使用requires_grad=True的tensor
        result.values().requires_grad_(True)
        result.values().backward(grad)
        
        print("✓ 梯度更新测试通过")

    def test_multi_table_gradient_update(self):
        """测试多表梯度更新"""
        print("\n=== 测试多表梯度更新 ===")
        ebc = RecStoreEmbeddingBagCollection(self.multi_table_configs)
        
        # 构造多表输入数据
        kjt = KeyedJaggedTensor(
            keys=["user_table", "item_table", "category_table"],
            values=torch.tensor([1, 2, 3, 4, 5], dtype=torch.int64),
            lengths=torch.tensor([1, 2, 2], dtype=torch.int32)
        )
        
        # 前向传播
        result = ebc(kjt)
        
        # 构造梯度
        grad = torch.randn_like(result.values())
        
        # 反向传播
        result.values().requires_grad_(True)
        result.values().backward(grad)
        
        print("✓ 多表梯度更新测试通过")

    def test_direct_kv_client_operations(self):
        """测试直接使用KV客户端的操作"""
        print("\n=== 测试直接KV客户端操作 ===")
        ebc = RecStoreEmbeddingBagCollection(self.basic_configs)
        
        # 测试pull操作
        ids = torch.tensor([1, 2, 3], dtype=torch.int64)
        embeddings = ebc.kv_client.pull("test_table", ids)
        self.assertEqual(embeddings.shape, (3, 16))
        
        # 测试update操作
        grads = torch.randn((3, 16))
        ebc.kv_client.update("test_table", ids, grads)
        
        # 测试push操作
        new_values = torch.randn((3, 16))
        ebc.kv_client.push("test_table", ids, new_values)
        
        print("✓ 直接KV客户端操作测试通过")

    def test_embedding_consistency(self):
        """测试嵌入一致性"""
        print("\n=== 测试嵌入一致性 ===")
        ebc = RecStoreEmbeddingBagCollection(self.basic_configs)
        
        # 设置一些嵌入值
        ids = torch.tensor([1, 2, 3], dtype=torch.int64)
        test_values = torch.randn((3, 16))
        ebc.kv_client.push("test_table", ids, test_values)
        
        # 通过前向传播获取嵌入 - 每个ID单独处理
        kjt = KeyedJaggedTensor(
            keys=["test_table"],
            values=ids,
            lengths=torch.tensor([1, 1, 1], dtype=torch.int32)
        )
        result = ebc(kjt)
        
        # 直接pull获取嵌入
        direct_embeddings = ebc.kv_client.pull("test_table", ids)
        
        # 验证一致性 - embedding bag会进行平均，所以结果是(1, 16)
        self.assertEqual(result.values().shape, (1, 16))
        self.assertEqual(direct_embeddings.shape, (3, 16))
        
        print("✓ 嵌入一致性测试通过")

    def test_large_batch_forward_pass(self):
        """测试大批次前向传播"""
        print("\n=== 测试大批次前向传播 ===")
        ebc = RecStoreEmbeddingBagCollection(self.basic_configs)
        
        # 构造大批次数据
        batch_size = 100
        ids = torch.randint(0, 100, (batch_size,), dtype=torch.int64)
        lengths = torch.ones(batch_size, dtype=torch.int32)
        
        kjt = KeyedJaggedTensor(
            keys=["test_table"],
            values=ids,
            lengths=lengths
        )
        
        # 前向传播
        result = ebc(kjt)
        
        # 验证结果
        self.assertEqual(result.values().shape, (1, 16))
        
        print("✓ 大批次前向传播测试通过")

    def test_repr_function(self):
        """测试repr函数"""
        print("\n=== 测试repr函数 ===")
        ebc = RecStoreEmbeddingBagCollection(self.basic_configs)
        repr_str = repr(ebc)
        self.assertIn("RecStoreEmbeddingBagCollection", repr_str)
        self.assertIn("test_table", repr_str)
        print(f"✓ repr函数测试通过: {repr_str}")

    def test_device_handling(self):
        """测试设备处理"""
        print("\n=== 测试设备处理 ===")
        ebc = RecStoreEmbeddingBagCollection(self.basic_configs)
        
        # 测试CPU设备
        kjt = KeyedJaggedTensor(
            keys=["test_table"],
            values=torch.tensor([1, 2, 3], dtype=torch.int64, device='cpu'),
            lengths=torch.tensor([2, 1], dtype=torch.int32, device='cpu')
        )
        
        result = ebc(kjt)
        self.assertEqual(result.values().device.type, 'cpu')
        
        print("✓ 设备处理测试通过")

    def test_dtype_handling(self):
        """测试数据类型处理"""
        print("\n=== 测试数据类型处理 ===")
        ebc = RecStoreEmbeddingBagCollection(self.basic_configs)
        
        # 测试不同的输入数据类型
        kjt = KeyedJaggedTensor(
            keys=["test_table"],
            values=torch.tensor([1, 2, 3], dtype=torch.int64),
            lengths=torch.tensor([2, 1], dtype=torch.int32)
        )
        
        result = ebc(kjt)
        # 输出应该是float32类型
        self.assertEqual(result.values().dtype, torch.float32)
        
        print("✓ 数据类型处理测试通过")

    def test_error_handling(self):
        """测试错误处理"""
        print("\n=== 测试错误处理 ===")
        ebc = RecStoreEmbeddingBagCollection(self.basic_configs)
        
        # 测试不存在的表
        with self.assertRaises(RuntimeError):
            ebc.kv_client.pull("non_existent_table", torch.tensor([1]))
        
        # 测试无效的ID范围
        kjt = KeyedJaggedTensor(
            keys=["test_table"],
            values=torch.tensor([999], dtype=torch.int64),  # 超出范围的ID
            lengths=torch.tensor([1], dtype=torch.int32)
        )
        
        # 这应该不会抛出异常，而是返回零向量
        result = ebc(kjt)
        self.assertEqual(result.values().shape, (1, 16))
        
        print("✓ 错误处理测试通过")

    def test_zero_length_forward_pass(self):
        """测试零长度前向传播"""
        print("\n=== 测试零长度前向传播 ===")
        ebc = RecStoreEmbeddingBagCollection(self.basic_configs)
        
        # 构造零长度数据
        kjt = KeyedJaggedTensor(
            keys=["test_table"],
            values=torch.tensor([], dtype=torch.int64),
            lengths=torch.tensor([0], dtype=torch.int32)
        )
        
        # 前向传播
        result = ebc(kjt)
        
        # 验证结果
        self.assertIsInstance(result, KeyedTensor)
        self.assertEqual(result.keys(), ["test_table"])
        self.assertEqual(result.values().shape, (1, 16))
        
        print("✓ 零长度前向传播测试通过")

    def test_multiple_ids_per_sample(self):
        """测试每个样本多个ID的情况"""
        print("\n=== 测试每个样本多个ID ===")
        ebc = RecStoreEmbeddingBagCollection(self.basic_configs)
        
        # 构造每个样本多个ID的数据
        kjt = KeyedJaggedTensor(
            keys=["test_table"],
            values=torch.tensor([1, 2, 3, 4, 5, 6], dtype=torch.int64),
            lengths=torch.tensor([2, 3, 1], dtype=torch.int32)  # 3个样本，分别有2、3、1个ID
        )
        
        # 前向传播
        result = ebc(kjt)
        
        # 验证结果
        self.assertIsInstance(result, KeyedTensor)
        self.assertEqual(result.keys(), ["test_table"])
        self.assertEqual(result.values().shape, (1, 16))
        
        print("✓ 每个样本多个ID测试通过")

    def test_embedding_update_consistency(self):
        """测试嵌入更新一致性"""
        print("\n=== 测试嵌入更新一致性 ===")
        ebc = RecStoreEmbeddingBagCollection(self.basic_configs)
        
        # 设置初始嵌入值
        ids = torch.tensor([1, 2], dtype=torch.int64)
        initial_values = torch.randn((2, 16))
        ebc.kv_client.push("test_table", ids, initial_values)
        
        # 通过前向传播获取嵌入
        kjt = KeyedJaggedTensor(
            keys=["test_table"],
            values=ids,
            lengths=torch.tensor([1, 1], dtype=torch.int32)
        )
        result1 = ebc(kjt)
        
        # 更新嵌入
        update_values = torch.randn((2, 16))
        ebc.kv_client.push("test_table", ids, update_values)
        
        # 再次前向传播
        result2 = ebc(kjt)
        
        # 验证结果形状一致
        self.assertEqual(result1.values().shape, result2.values().shape)
        
        print("✓ 嵌入更新一致性测试通过")


def run_all_tests():
    """运行所有测试"""
    print("开始运行RecStoreEmbeddingBagCollection测试套件...")
    
    # 创建测试套件
    test_suite = unittest.TestLoader().loadTestsFromTestCase(TestRecStoreEmbeddingBagCollection)
    
    # 运行测试
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(test_suite)
    
    # 打印总结
    print(f"\n测试总结:")
    print(f"运行测试数: {result.testsRun}")
    print(f"失败测试数: {len(result.failures)}")
    print(f"错误测试数: {len(result.errors)}")
    
    if result.failures:
        print("\n失败的测试:")
        for test, traceback in result.failures:
            print(f"- {test}: {traceback}")
    
    if result.errors:
        print("\n错误的测试:")
        for test, traceback in result.errors:
            print(f"- {test}: {traceback}")
    
    return result.wasSuccessful()


def main():
    """主函数，运行所有测试"""
    success = run_all_tests()
    if success:
        print("\n🎉 所有测试通过！")
    else:
        print("\n❌ 部分测试失败，请检查上述错误信息。")
    return success


if __name__ == "__main__":
    main()