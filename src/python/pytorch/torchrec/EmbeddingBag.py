import torch
import logging
import torch.nn.functional as F
from torch.autograd import Function
from typing import List, Dict, Any, Tuple
from torchrec.sparse.jagged_tensor import KeyedJaggedTensor, KeyedTensor
from torchrec.modules.embedding_configs import EmbeddingBagConfig
from ..recstore.KVClient import get_kv_client, RecStoreClient
from torch.profiler import record_function

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
        with record_function("recstore_ebc_function.forward"):
            ctx.save_for_backward(features_values, features_lengths)
            ctx.module = module
            ctx.feature_keys = feature_keys

            config = module.embedding_bag_configs()[0]

            with record_function("kv_pull"):
                all_embeddings = module.kv_client.pull(name=config.name, ids=features_values)

            all_embeddings.requires_grad = True

            local_indices = torch.arange(
                len(features_values), device=features_values.device, dtype=torch.long
            )

            with record_function("compute_offsets"):
                offsets = torch.cat(
                    [torch.tensor([0], device=features_lengths.device), torch.cumsum(features_lengths, 0)[:-1]]
                )

            with record_function("embedding_bag_pool"):
                pooled_output = F.embedding_bag(
                    input=local_indices,
                    weight=all_embeddings,
                    offsets=offsets,
                    mode="sum",
                    sparse=False,
                )

            batch_size = features_lengths.numel() // len(module.feature_keys)
            embedding_dim = config.embedding_dim

            with record_function("reshape_output"):
                return pooled_output.view(batch_size, len(feature_keys), embedding_dim)

    @staticmethod
    def backward(
        ctx, grad_output_values: torch.Tensor
    ) -> Tuple[None, None, None, None]:
        with record_function("recstore_ebc_function.backward"):
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

    # Note: updates are performed during backward in grad_hook; apply_accumulated_updates is intentionally removed

    def forward(self, features: KeyedJaggedTensor) -> KeyedTensor:
        with record_function("recstore_ebc.forward"):
            pooled_embs_list = []

            for key in features.keys():
                with record_function(f"recstore_ebc.per_feature[{key}]"):
                    config_name = self._config_names[key]
                    kjt_per_feature = features[key]
                    values = kjt_per_feature.values()
                    lengths = kjt_per_feature.lengths()

                    if values.numel() == 0:
                        config = next(c for c in self._embedding_bag_configs if key in c.feature_names)
                        pooled_embs = torch.zeros(len(lengths), config.embedding_dim, device=features.device(), dtype=torch.float32)
                    else:
                        with record_function("kv_pull"):
                            all_embeddings = self.kv_client.pull(name=config_name, ids=values)
                        all_embeddings.requires_grad_()

                        def grad_hook(grad, name=config_name, ids=values):
                            with record_function("recstore_ebc.backward.grad_hook"):
                                # Aggregate on-device to reduce work, then do KV ops on CPU to match backend support
                                ids_device = ids.detach().to(torch.int64)
                                grad_device = grad.detach().to(torch.float32)
                                if ids_device.numel() == 0:
                                    return
                                with record_function("unique_and_sum"):
                                    unique_ids, inverse = torch.unique(ids_device, return_inverse=True)
                                    grad_sum = torch.zeros((unique_ids.size(0), grad_device.size(1)), dtype=grad_device.dtype, device=grad_device.device)
                                    grad_sum.index_add_(0, inverse, grad_device)
                                with record_function("to_cpu_for_kv_ops"):
                                    unique_ids_cpu = unique_ids.to(dtype=torch.int64, device="cpu", non_blocking=False).contiguous()
                                    grad_sum_cpu = grad_sum.to(dtype=torch.float32, device="cpu", non_blocking=False).contiguous()
                                with record_function("kv_pull_current"):
                                    current = self.kv_client.pull(name=name, ids=unique_ids_cpu)
                                with record_function("sgd_update"):
                                    if current.dtype != grad_sum_cpu.dtype:
                                        current = current.to(dtype=grad_sum_cpu.dtype)
                                    updated = current - self._lr * grad_sum_cpu
                                with record_function("kv_push_update"):
                                    self.kv_client.push(name=name, ids=unique_ids_cpu, data=updated.contiguous())
                        all_embeddings.register_hook(grad_hook)

                        local_indices = torch.arange(len(values), device=values.device, dtype=torch.long)
                        offsets = torch.cat([torch.tensor([0], device=lengths.device), torch.cumsum(lengths, 0)[:-1]])
                        with record_function("embedding_bag_pool"):
                            pooled_embs = F.embedding_bag(
                                input=local_indices,
                                weight=all_embeddings,
                                offsets=offsets,
                                mode="sum",
                                sparse=False,
                            )

                    pooled_embs_list.append(pooled_embs)

            with record_function("concat_embeddings"):
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
