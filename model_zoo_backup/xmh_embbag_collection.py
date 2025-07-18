import abc
from typing import Dict, List, Optional, Union

import torch
import torch.nn as nn
from torchrec.modules.embedding_configs import (
    DataType,
    EmbeddingBagConfig,
    EmbeddingConfig,
    pooling_type_to_str,
)
from torchrec.sparse.jagged_tensor import JaggedTensor, KeyedJaggedTensor, KeyedTensor
from torchrec.modules.embedding_modules import EmbeddingBagCollectionInterface

from torchrec.modules.embedding_modules import (
    get_embedding_names_by_table,
)

class CustomEmbeddingBagCollection(EmbeddingBagCollectionInterface):
    """
    EmbeddingBagCollection represents a collection of pooled embeddings (`EmbeddingBags`).

    It processes sparse data in the form of `KeyedJaggedTensor` with values of the form
    [F X B X L] where:

    * F: features (keys)
    * B: batch size
    * L: length of sparse features (jagged)

    and outputs a `KeyedTensor` with values of the form [B * (F * D)] where:

    * F: features (keys)
    * D: each feature's (key's) embedding dimension
    * B: batch size

    Args:
        tables (List[EmbeddingBagConfig]): list of embedding tables.
        is_weighted (bool): whether input `KeyedJaggedTensor` is weighted.
        device (Optional[torch.device]): default compute device.

    Example::

        table_0 = EmbeddingBagConfig(
            name="t1", embedding_dim=3, num_embeddings=10, feature_names=["f1"]
        )
        table_1 = EmbeddingBagConfig(
            name="t2", embedding_dim=4, num_embeddings=10, feature_names=["f2"]
        )

        ebc = EmbeddingBagCollection(tables=[table_0, table_1])

        #        0       1        2  <-- batch
        # "f1"   [0,1] None    [2]
        # "f2"   [3]    [4]    [5,6,7]
        #  ^
        # feature

        features = KeyedJaggedTensor(
            keys=["f1", "f2"],
            values=torch.tensor([0, 1, 2, 3, 4, 5, 6, 7]),
            offsets=torch.tensor([0, 2, 2, 3, 4, 5, 8]),
        )

        pooled_embeddings = ebc(features)
        print(pooled_embeddings.values())
        tensor([[-0.8899, -0.1342, -1.9060, -0.0905, -0.2814, -0.9369, -0.7783],
            [ 0.0000,  0.0000,  0.0000,  0.1598,  0.0695,  1.3265, -0.1011],
            [-0.4256, -1.1846, -2.1648, -1.0893,  0.3590, -1.9784, -0.7681]],
            grad_fn=<CatBackward0>)
        print(pooled_embeddings.keys())
        ['f1', 'f2']
        print(pooled_embeddings.offset_per_key())
        tensor([0, 3, 7])
    """

    def __init__(
        self,
        tables: List[EmbeddingBagConfig],
        is_weighted: bool = False,
        device: Optional[torch.device] = None,
    ) -> None:
        self.cached_ratio = 0.05

        super().__init__()
        torch._C._log_api_usage_once(f"torchrec.modules.{self.__class__.__name__}")
        self._is_weighted = is_weighted

        self.embedding_bags: nn.ModuleDict = nn.ModuleDict()
        self.cached_embedding_bags: nn.ModuleDict = nn.ModuleDict()

        
        

        self._embedding_bag_configs = tables
        self._lengths_per_embedding: List[int] = []
        self._device: torch.device = (
            device if device is not None else torch.device("cpu")
        )

        table_names = set()
        for embedding_config in tables:
            if embedding_config.name in table_names:
                raise ValueError(f"Duplicate table name {embedding_config.name}")
            table_names.add(embedding_config.name)
            dtype = (
                torch.float32
                if embedding_config.data_type == DataType.FP32
                else torch.float16
            )
            self.embedding_bags[embedding_config.name] = nn.EmbeddingBag(
                num_embeddings=embedding_config.num_embeddings,
                embedding_dim=embedding_config.embedding_dim,
                mode=pooling_type_to_str(embedding_config.pooling),
                device=torch.device("cpu"),
                include_last_offset=True,
                dtype=dtype,
            )

            self.cached_embedding_bags[embedding_config.name] = None

            # self.cached_embedding_bags[embedding_config.name] = nn.EmbeddingBag(
            #     num_embeddings=int(embedding_config.num_embeddings * self.cached_ratio),
            #     embedding_dim=embedding_config.embedding_dim,
            #     mode=pooling_type_to_str(embedding_config.pooling),
            #     device=torch.device("cuda"),
            #     include_last_offset=True,
            #     dtype=dtype,
            # )
            
            
            if not embedding_config.feature_names:
                embedding_config.feature_names = [embedding_config.name]
            self._lengths_per_embedding.extend(
                len(embedding_config.feature_names) * [embedding_config.embedding_dim]
            )

        self._embedding_names: List[str] = [
            embedding
            for embeddings in get_embedding_names_by_table(tables)
            for embedding in embeddings
        ]
        self._feature_names: List[List[str]] = [table.feature_names for table in tables]

    def forward(self, features: KeyedJaggedTensor) -> KeyedTensor:
        """
        Args:
            features (KeyedJaggedTensor): KJT of form [F X B X L].

        Returns:
            KeyedTensor
        """

        pooled_embeddings: List[torch.Tensor] = []

        feature_dict = features.to_dict()
        for i, (key, embedding_bag) in enumerate(self.embedding_bags.items()):
            for feature_name in self._feature_names[i]:
                f = feature_dict[feature_name]

                # f is JaggedTensor

                # print("feature_name", feature_name)
                # print("f", f)
                # print("value", f.values())
                # print("offset", f.offsets())

               
                cached_embedding_bag = self.cached_embedding_bags[key] 
                
                res = embedding_bag(
                    input=f.values(),
                    offsets=f.offsets(),
                    per_sample_weights=f.weights() if self._is_weighted else None,
                ).float()



                pooled_embeddings.append(res)
        data = torch.cat(pooled_embeddings, dim=1)
        return KeyedTensor(
            keys=self._embedding_names,
            values=data,
            length_per_key=self._lengths_per_embedding,
        )

    def is_weighted(self) -> bool:
        return self._is_weighted

    def embedding_bag_configs(self) -> List[EmbeddingBagConfig]:
        return self._embedding_bag_configs

    @property
    def device(self) -> torch.device:
        return self._device