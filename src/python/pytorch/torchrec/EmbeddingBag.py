import torch
import logging
import torch.nn.functional as F
from torch.autograd import Function
from typing import List, Dict, Any, Tuple
from torchrec.sparse.jagged_tensor import KeyedJaggedTensor, KeyedTensor
from torchrec.modules.embedding_configs import EmbeddingBagConfig
from ..recstore.KVClient import get_kv_client, RecStoreClient

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)


class _RecStoreEBCFunction(Function):
    @staticmethod
    def forward(
        ctx,
        module: "RecStoreEmbeddingBagCollection",
        feature_keys: List[str],
        features_values: torch.Tensor,
        features_lengths: torch.Tensor,
    ) -> torch.Tensor:
        ctx.save_for_backward(features_values, features_lengths)
        ctx.module = module
        ctx.feature_keys = feature_keys

        config = module.embedding_bag_configs()[0]
        
        all_embeddings = module.kv_client.pull(name=config.name, ids=features_values)

        all_embeddings.requires_grad = True

        local_indices = torch.arange(
            len(features_values), device=features_values.device, dtype=torch.long
        )

        offsets = torch.cat(
            [torch.tensor([0], device=features_lengths.device), torch.cumsum(features_lengths, 0)[:-1]]
        )

        pooled_output = F.embedding_bag(
            input=local_indices,
            weight=all_embeddings,
            offsets=offsets,
            mode="sum",
            sparse=False,
        )

        batch_size = features_lengths.numel() // len(module.feature_keys)
        embedding_dim = config.embedding_dim
        
        return pooled_output.view(batch_size, len(feature_keys), embedding_dim)

    @staticmethod
    def backward(
        ctx, grad_output_values: torch.Tensor
    ) -> Tuple[None, None, None, None]:
        features_values, features_lengths = ctx.saved_tensors
        module: "RecStoreEmbeddingBagCollection" = ctx.module
        feature_keys: List[str] = ctx.feature_keys

        batch_size = features_lengths.numel() // len(feature_keys)
        grad_output_reshaped = grad_output_values.view(batch_size, len(feature_keys), -1)

        lengths_cpu = features_lengths.cpu()
        values_cpu = features_values.cpu()
        
        offsets = torch.cat([torch.tensor([0]), torch.cumsum(lengths_cpu, 0)])

        for i, key in enumerate(feature_keys):
            config_name = module._config_names[key]
            
            for sample_idx in range(batch_size):
                feature_in_batch_idx = sample_idx * len(feature_keys) + i
                start, end = offsets[feature_in_batch_idx], offsets[feature_in_batch_idx + 1]
                
                num_items_in_bag = end - start
                if num_items_in_bag > 0:
                    ids_to_update = values_cpu[start:end]
                    grad_for_bag = grad_output_reshaped[sample_idx, i]
                    scaled_grad = grad_for_bag
                    
                    grads_to_trace = scaled_grad.unsqueeze(0).expand(num_items_in_bag, -1)
                    
                    module._trace.append(
                        (config_name, ids_to_update.detach(), grads_to_trace.detach())
                    )
        return None, None, None, None


class RecStoreEmbeddingBagCollection(torch.nn.Module):
    def __init__(self, embedding_bag_configs: List[Dict[str, Any]], lr: float = 0.01):
        super().__init__()
        self._embedding_bag_configs = [
            EmbeddingBagConfig(**c) for c in embedding_bag_configs
        ]
        self.kv_client: RecStoreClient = get_kv_client()
        self._lr = lr
        
        self.feature_keys: List[str] = []
        self._config_names: Dict[str, str] = {}
        self._embedding_dims: List[int] = [] 
        for c in self._embedding_bag_configs:
            for feature_name in c.feature_names:
                self.feature_keys.append(feature_name)
                self._config_names[feature_name] = c.name
                self._embedding_dims.append(c.embedding_dim)

        self._trace = []

        for config in self._embedding_bag_configs:
            self.kv_client.init_data(
                name=config.name,
                shape=(config.num_embeddings, config.embedding_dim),
                dtype=torch.float32,
            )

    def embedding_bag_configs(self):
        return self._embedding_bag_configs

    def reset_trace(self):
        self._trace = []

    def forward(self, features: KeyedJaggedTensor) -> KeyedTensor:
        pooled_embs_list = []
        
        for key in features.keys():
            config_name = self._config_names[key]
            kjt_per_feature = features[key]
            values = kjt_per_feature.values()
            lengths = kjt_per_feature.lengths()

            if values.numel() == 0:
                config = next(c for c in self._embedding_bag_configs if key in c.feature_names)
                pooled_embs = torch.zeros(len(lengths), config.embedding_dim, device=features.device(), dtype=torch.float32)
            else:
                all_embeddings = self.kv_client.pull(name=config_name, ids=values)
                all_embeddings.requires_grad_()

                def grad_hook(grad, name=config_name, ids=values):
                    ids_cpu = ids.detach().to(torch.int64).cpu()
                    grad_cpu = grad.detach().to(torch.float32).cpu()
                    if ids_cpu.numel() == 0:
                        return
                    unique_ids, inverse = torch.unique(ids_cpu, return_inverse=True)
                    grad_sum = torch.zeros((unique_ids.size(0), grad_cpu.size(1)), dtype=grad_cpu.dtype)
                    grad_sum.index_add_(0, inverse, grad_cpu)
                    current = self.kv_client.pull(name=name, ids=unique_ids)
                    updated = current - self._lr * grad_sum
                    self.kv_client.push(name=name, ids=unique_ids, data=updated)
                    self._trace.append((name, unique_ids, grad_sum))
                all_embeddings.register_hook(grad_hook)

                local_indices = torch.arange(len(values), device=values.device, dtype=torch.long)
                offsets = torch.cat([torch.tensor([0], device=lengths.device), torch.cumsum(lengths, 0)[:-1]])
                pooled_embs = F.embedding_bag(
                    input=local_indices,
                    weight=all_embeddings,
                    offsets=offsets,
                    mode="sum",
                    sparse=False,
                )
            
            pooled_embs_list.append(pooled_embs)
        
        concatenated_embs = torch.cat(pooled_embs_list, dim=1)


        length_per_key = [
            next(c.embedding_dim for c in self._embedding_bag_configs if key in c.feature_names)
            for key in features.keys()
        ]

        return KeyedTensor(
            keys=features.keys(),
            values=concatenated_embs,
            length_per_key=length_per_key,
        )

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(tables={self.feature_keys})"
