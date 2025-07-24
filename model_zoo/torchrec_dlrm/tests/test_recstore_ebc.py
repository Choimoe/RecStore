import torch
import os
import sys
from torchrec.sparse.jagged_tensor import KeyedJaggedTensor
RECSTORE_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../../src'))
if RECSTORE_PATH not in sys.path:
    sys.path.insert(0, RECSTORE_PATH)

from python.pytorch.torchrec.EmbeddingBag import RecStoreEmbeddingBagCollection

def main():
    eb_configs = [
        {
            "name": "test_table",
            "num_embeddings": 100,
            "embedding_dim": 16,
            "feature_names": ["test_feature"]
        }
    ]
    ebc = RecStoreEmbeddingBagCollection(eb_configs)

    # 构造一个 KeyedJaggedTensor，模拟 batch=2，每个样本1~2个id
    # 假设 feature_keys = ["test_table"]
    # ids: [1, 2, 3]
    # lengths: [2, 1]  # 第一个样本2个id，第二个样本1个id
    kjt = KeyedJaggedTensor(
        keys=["test_table"],
        values=torch.tensor([1, 2, 3], dtype=torch.int64),
        lengths=torch.tensor([2, 1], dtype=torch.int32)
    )

    kt = ebc(kjt)
    print("Embedding lookup result (KeyedTensor):")
    print("keys:", kt.keys())
    print("values shape:", kt.values().shape)
    print("values:", kt.values())
    print("lengths:", kt.length_per_key())

    ids = torch.tensor([1, 2, 3], dtype=torch.int64)
    grad = torch.randn((3, 16))
    ebc.kv_client.update("test_table", ids, grad)
    print("Update called successfully.")

if __name__ == "__main__":
    main()