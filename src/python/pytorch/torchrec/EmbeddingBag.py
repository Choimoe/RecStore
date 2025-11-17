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
    def __init__(self, embedding_bag_configs: List[Dict[str, Any]], lr: float = 0.01,
                 enable_fusion: bool = True, fusion_k: int = 30):
        super().__init__()
        self._embedding_bag_configs = [
            EmbeddingBagConfig(**c) for c in embedding_bag_configs
        ]
        self.kv_client: RecStoreClient = get_kv_client()
        self._lr = lr
        self._enable_fusion = enable_fusion
        self._fusion_k = fusion_k
        
        self.feature_keys: List[str] = []
        self._config_names: Dict[str, str] = {}
        self._embedding_dims: List[int] = [] 
        for c in self._embedding_bag_configs:
            for feature_name in c.feature_names:
                self.feature_keys.append(feature_name)
                self._config_names[feature_name] = c.name
                self._embedding_dims.append(c.embedding_dim)

        # Use the first table's config as the "master" to obtain embedding_dim
        # for fused reads/writes (backend only needs dim; name is not used).
        self._master_config = self._embedding_bag_configs[0] if len(self._embedding_bag_configs) > 0 else None

        self._trace = []
        self._prefetch_handles: Dict[str, int] = {}
        # Fused prefetch handle (single handle for all fused IDs)
        self._fused_prefetch_handle: int | None = None
        self._fused_prefetch_num_ids: int = 0
        self._fused_prefetch_issue_ts: float | None = None
        # Prefetch performance stats
        self._prefetch_issue_ts: Dict[int, float] = {}  # handle -> issue time
        self._prefetch_sizes: Dict[int, int] = {}       # handle -> number of ids
        self._prefetch_wait_latencies: List[float] = []
        self._prefetch_issue_latencies: List[float] = []  # currently producer-side if provided
        self._prefetch_total_ids: int = 0

        for idx, config in enumerate(self._embedding_bag_configs):
            base_offset = (idx << self._fusion_k) if self._enable_fusion else 0
            self.kv_client.init_data(
                name=config.name,
                shape=(config.num_embeddings, config.embedding_dim),
                dtype=torch.float32,
                base_offset=base_offset,
            )

    def embedding_bag_configs(self):
        return self._embedding_bag_configs

    def reset_trace(self):
        self._trace = []

    def set_fusion(self, enabled: bool):
        """Enable/disable fused embedding path at runtime."""
        self._enable_fusion = bool(enabled)

    def set_prefetch_handles(self, handles: Dict[str, Any]):
        """Set prefetch handles plus optional stats metadata.

        Accepts: { feature_key: handle } OR { feature_key: (handle, num_ids, issue_ts) }
        """
        import time
        parsed: Dict[str, int] = {}
        now = time.time()
        for k, v in handles.items():
            if isinstance(v, tuple):
                if len(v) == 3:
                    h, num_ids, t_issue = v
                elif len(v) == 2:
                    h, num_ids = v; t_issue = now
                else:
                    h = v[0]; num_ids = 0; t_issue = now
                parsed[k] = int(h)
                self._prefetch_issue_ts[int(h)] = float(t_issue)
                self._prefetch_sizes[int(h)] = int(num_ids)
                self._prefetch_total_ids += int(num_ids)
            else:
                parsed[k] = int(v)
                # Unknown size, leave stats partial
        self._prefetch_handles = parsed

    def set_fused_prefetch_handle(self, handle: int, num_ids: int | None = None, issue_ts: float | None = None):
        """Set a single fused prefetch handle for the upcoming forward.

        Optionally record stats: number of ids and issue timestamp for latency accounting.
        """
        import time
        self._fused_prefetch_handle = int(handle)
        self._fused_prefetch_num_ids = int(num_ids) if num_ids is not None else 0
        self._fused_prefetch_issue_ts = float(issue_ts) if issue_ts is not None else time.time()
        # Track in shared stats maps for unified reporting
        self._prefetch_issue_ts[self._fused_prefetch_handle] = self._fused_prefetch_issue_ts
        if self._fused_prefetch_num_ids:
            self._prefetch_sizes[self._fused_prefetch_handle] = self._fused_prefetch_num_ids
            self._prefetch_total_ids += self._fused_prefetch_num_ids

    def issue_fused_prefetch(self, features: KeyedJaggedTensor) -> int:
        """Compute fused global IDs for the given features and issue a single prefetch.

        Returns the prefetch handle. Use together with the next forward call.
        """
        import time
        if not self._enable_fusion or self._master_config is None:
            raise RuntimeError("Fused prefetch requires fusion enabled and a valid master config.")

        keys_in_batch = list(features.keys())
        fused_values_list: List[torch.Tensor] = []
        device = features.device()
        for key in keys_in_batch:
            kjt_per_feature = features[key]
            values = kjt_per_feature.values()
            if values.dtype != torch.int64:
                values = values.to(torch.int64)
            if values.numel() > 0:
                table_idx = next(i for i, c in enumerate(self._embedding_bag_configs) if key in c.feature_names)
                prefix = (table_idx << self._fusion_k)
                fused_values = values + prefix
                fused_values_list.append(fused_values)
        fused_values_all = torch.cat(fused_values_list, dim=0) if len(fused_values_list) > 0 else torch.empty((0,), dtype=torch.int64, device=device)
        handle = self.kv_client.prefetch(fused_values_all)
        # record
        self.set_fused_prefetch_handle(handle, num_ids=int(fused_values_all.numel()), issue_ts=time.time())
        return handle

    def forward(self, features: KeyedJaggedTensor) -> KeyedTensor:
        # Determine if we can enable fused single-call path safely
        keys_in_batch = list(features.keys())
        dims_this_batch: List[int] = [
            next(c.embedding_dim for c in self._embedding_bag_configs if k in c.feature_names)
            for k in keys_in_batch
        ]
        can_uniform_dim = all(d == dims_this_batch[0] for d in dims_this_batch) if len(dims_this_batch) > 0 else False

        use_fused = (
            self._enable_fusion
            and self._master_config is not None
            and can_uniform_dim
        )

        if use_fused:
            # Build fused global IDs with bit-prefix encoding: fused_id = (table_id << k) | id
            fused_values_list: List[torch.Tensor] = []
            lengths_total_list: List[torch.Tensor] = []
            compute_device = features.device()  # target device (cuda or cpu)

            for key in keys_in_batch:
                kjt_per_feature = features[key]
                values = kjt_per_feature.values()
                lengths = kjt_per_feature.lengths()
                # ensure dtype int64
                if values.dtype != torch.int64:
                    values = values.to(torch.int64)

                if values.numel() > 0:
                    table_idx = next(i for i, c in enumerate(self._embedding_bag_configs) if key in c.feature_names)
                    prefix = (table_idx << self._fusion_k)
                    fused_values = values + prefix
                    fused_values_list.append(fused_values)
                # Even if empty, we should keep lengths for offsets construction
                lengths_total_list.append(lengths)

            if len(fused_values_list) > 0:
                fused_values_all = torch.cat(fused_values_list, dim=0).to(compute_device)
            else:
                fused_values_all = torch.empty((0,), dtype=torch.int64, device=compute_device)
            # CPU copy for KV pull
            cpu_ids = fused_values_all.to('cpu')
            if not cpu_ids.is_contiguous():
                cpu_ids = cpu_ids.contiguous()

            lengths_total = torch.cat(lengths_total_list, dim=0) if len(lengths_total_list) > 0 else torch.empty((0,), dtype=torch.int32, device=compute_device)

            # Obtain embeddings: prefer fused prefetch handle; else merge per-feature prefetch; else single pull
            all_embeddings: torch.Tensor
            used_fused_prefetch = False
            if self._fused_prefetch_handle is not None:
                import time
                t_wait_start = time.time()
                all_embeddings = self.kv_client.wait_and_get(self._fused_prefetch_handle, self._master_config.embedding_dim, device=compute_device)
                t_wait_end = time.time()
                # stats
                wait_latency = t_wait_end - t_wait_start
                self._prefetch_wait_latencies.append(wait_latency)
                issue_latency = t_wait_start - (self._fused_prefetch_issue_ts or t_wait_start)
                self._prefetch_issue_latencies.append(issue_latency)
                used_fused_prefetch = True
                # Validate size matches
                if all_embeddings.size(0) != fused_values_all.numel():
                    logging.warning(
                        f"[EBC] Fused prefetch result size mismatch: got {all_embeddings.size(0)}, expected {fused_values_all.numel()}, falling back to pull."
                    )
                    all_embeddings = self.kv_client.pull(name=self._master_config.name, ids=cpu_ids)
                    if compute_device.type == 'cuda':
                        all_embeddings = all_embeddings.to(compute_device)
                    used_fused_prefetch = False
            elif len(self._prefetch_handles) > 0:
                # Gather per-feature prefetched embeddings in the same order
                per_feature_embs: List[torch.Tensor] = []
                for key in keys_in_batch:
                    kjt_per_feature = features[key]
                    values = kjt_per_feature.values()
                    if key in self._prefetch_handles:
                        import time
                        handle = self._prefetch_handles.pop(key)
                        config = next(c for c in self._embedding_bag_configs if key in c.feature_names)
                        t_wait_start = time.time()
                        emb = self.kv_client.wait_and_get(handle, config.embedding_dim, device=values.device)
                        t_wait_end = time.time()
                        if handle in self._prefetch_issue_ts:
                            wait_latency = t_wait_end - t_wait_start
                            issue_latency = t_wait_start - self._prefetch_issue_ts.get(handle, t_wait_start)
                            self._prefetch_wait_latencies.append(wait_latency)
                            self._prefetch_issue_latencies.append(issue_latency)
                        if emb.size(0) != values.numel():
                            logging.warning(
                                f"[EBC] Prefetch result size mismatch for feature '{key}': got {emb.size(0)}, expected {values.numel()}, falling back to pull."
                            )
                            table_idx = next(i for i, c in enumerate(self._embedding_bag_configs) if key in c.feature_names)
                            cpu_ids_local = (values.to(torch.int64) + (table_idx << self._fusion_k)).to('cpu')
                            emb = self.kv_client.pull(name=self._master_config.name, ids=cpu_ids_local)
                            if values.device.type == 'cuda':
                                emb = emb.to(values.device)
                        per_feature_embs.append(emb)
                    else:
                        # No prefetch for this feature, do a synchronous pull with fused IDs for that feature only
                        if values.numel() == 0:
                            config = next(c for c in self._embedding_bag_configs if key in c.feature_names)
                            per_feature_embs.append(torch.empty((0, config.embedding_dim), device=features.device(), dtype=torch.float32))
                        else:
                            table_idx = next(i for i, c in enumerate(self._embedding_bag_configs) if key in c.feature_names)
                            prefix = (table_idx << self._fusion_k)
                            cpu_ids_local = (values.to(torch.int64) + prefix).to('cpu')
                            emb = self.kv_client.pull(name=self._master_config.name, ids=cpu_ids_local)
                            if values.device.type == 'cuda':
                                emb = emb.to(values.device)
                            per_feature_embs.append(emb)
                all_embeddings = torch.cat(per_feature_embs, dim=0) if len(per_feature_embs) > 0 else torch.empty((0, self._master_config.embedding_dim), device=features.device(), dtype=torch.float32)
            else:
                # Single pull for all fused IDs
                all_embeddings = self.kv_client.pull(name=self._master_config.name, ids=cpu_ids)
                if compute_device.type == 'cuda':
                    all_embeddings = all_embeddings.to(compute_device)
            all_embeddings.requires_grad_()

            def grad_hook_fused(grad, ids=fused_values_all, master_name=self._master_config.name):
                ids_cpu = ids.detach().to(torch.int64).cpu()
                grad_cpu = grad.detach().to(torch.float32).cpu()
                if ids_cpu.numel() == 0:
                    return
                unique_ids, inverse = torch.unique(ids_cpu, return_inverse=True)
                grad_sum = torch.zeros((unique_ids.size(0), grad_cpu.size(1)), dtype=grad_cpu.dtype)
                grad_sum.index_add_(0, inverse, grad_cpu)
                current = self.kv_client.pull(name=master_name, ids=unique_ids)
                updated = current - self._lr * grad_sum
                self.kv_client.push(name=master_name, ids=unique_ids, data=updated)
                self._trace.append((master_name, unique_ids, grad_sum))

            all_embeddings.register_hook(grad_hook_fused)

            # Pool across all bags (feature-major order)
            local_indices = torch.arange(len(fused_values_all), device=compute_device, dtype=torch.long)
            offsets = torch.cat([
                torch.tensor([0], device=compute_device),
                torch.cumsum(lengths_total, 0)[:-1] if lengths_total.numel() > 0 else torch.empty((0,), device=compute_device, dtype=lengths_total.dtype)
            ])
            pooled_total = F.embedding_bag(
                input=local_indices,
                weight=all_embeddings,
                offsets=offsets,
                mode="sum",
                sparse=False,
            )

            # Split back by feature (each has B_i bags = lengths.size(0))
            pooled_embs_list: List[torch.Tensor] = []
            split_sizes = [l.numel() for l in lengths_total_list]
            if sum(split_sizes) > 0:
                pieces = torch.split(pooled_total, split_sizes, dim=0)
            else:
                pieces = [torch.empty((0, dims_this_batch[0]), device=pooled_total.device, dtype=pooled_total.dtype) for _ in split_sizes]
            for piece in pieces:
                pooled_embs_list.append(piece)

            concatenated_embs = torch.cat(pooled_embs_list, dim=1) if len(pooled_embs_list) > 0 else torch.empty((0, dims_this_batch[0]), device=compute_device, dtype=torch.float32)

            length_per_key = dims_this_batch

            out = KeyedTensor(
                keys=keys_in_batch,
                values=concatenated_embs,
                length_per_key=length_per_key,
            )
            # Clear fused prefetch handle after consumption
            if used_fused_prefetch:
                self._fused_prefetch_handle = None
                self._fused_prefetch_issue_ts = None
                self._fused_prefetch_num_ids = 0
            return out

        # Fallback: per-feature path (original behavior)
        pooled_embs_list = []
        for key in keys_in_batch:
            config_name = self._config_names[key]
            kjt_per_feature = features[key]
            values = kjt_per_feature.values()
            lengths = kjt_per_feature.lengths()

            if values.numel() == 0:
                config = next(c for c in self._embedding_bag_configs if key in c.feature_names)
                pooled_embs = torch.zeros(len(lengths), config.embedding_dim, device=features.device(), dtype=torch.float32)
            else:
                # Prefer prefetched embeddings if provided for this feature
                ids_used = values  # default
                if key in self._prefetch_handles:
                    import time
                    handle = self._prefetch_handles.pop(key)
                    config = next(c for c in self._embedding_bag_configs if key in c.feature_names)
                    t_wait_start = time.time()
                    all_embeddings = self.kv_client.wait_and_get(handle, config.embedding_dim, device=values.device)
                    t_wait_end = time.time()
                    if handle in self._prefetch_issue_ts:
                        wait_latency = t_wait_end - t_wait_start
                        issue_latency = t_wait_start - self._prefetch_issue_ts.get(handle, t_wait_start)
                        self._prefetch_wait_latencies.append(wait_latency)
                        self._prefetch_issue_latencies.append(issue_latency)
                    # Fallback: if prefetch result size mismatches, do sync pull
                    if all_embeddings.size(0) != values.numel():
                        logging.warning(
                            f"[EBC] Prefetch result size mismatch for feature '{key}': got {all_embeddings.size(0)}, expected {values.numel()}, falling back to pull."
                        )
                        # If fusion was enabled at init, IDs in storage are prefixed; add offset here
                        if self._enable_fusion:
                            table_idx = next(i for i, c in enumerate(self._embedding_bag_configs) if key in c.feature_names)
                            prefix = (table_idx << self._fusion_k)
                            ids_used = values.to(torch.int64) + prefix
                            cpu_ids_local = ids_used.to('cpu')
                            all_embeddings = self.kv_client.pull(name=self._master_config.name, ids=cpu_ids_local)
                            if values.device.type == 'cuda':
                                all_embeddings = all_embeddings.to(values.device)
                        else:
                            ids_used = values
                            all_embeddings = self.kv_client.pull(name=config_name, ids=ids_used)
                            if values.device.type == 'cuda':
                                all_embeddings = all_embeddings.to(values.device)
                else:
                    if self._enable_fusion:
                        table_idx = next(i for i, c in enumerate(self._embedding_bag_configs) if key in c.feature_names)
                        prefix = (table_idx << self._fusion_k)
                        ids_used = values.to(torch.int64) + prefix
                        cpu_ids_local = ids_used.to('cpu')
                        all_embeddings = self.kv_client.pull(name=self._master_config.name, ids=cpu_ids_local)
                        if values.device.type == 'cuda':
                            all_embeddings = all_embeddings.to(values.device)
                    else:
                        ids_used = values
                        all_embeddings = self.kv_client.pull(name=config_name, ids=ids_used)
                        if values.device.type == 'cuda':
                            all_embeddings = all_embeddings.to(values.device)
                all_embeddings.requires_grad_()

                def grad_hook(grad, name=(self._master_config.name if self._enable_fusion else config_name), ids=ids_used):
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
            for key in keys_in_batch
        ]
        return KeyedTensor(
            keys=keys_in_batch,
            values=concatenated_embs,
            length_per_key=length_per_key,
        )

    def report_prefetch_stats(self, reset: bool = True) -> Dict[str, float]:
        import math
        n = len(self._prefetch_wait_latencies)
        if n == 0:
            stats = {
                "batches_prefetched": 0,
                "avg_wait_ms": 0.0,
                "avg_issue_to_wait_ms": 0.0,
                "total_prefetched_ids": self._prefetch_total_ids,
            }
        else:
            avg_wait = sum(self._prefetch_wait_latencies) / n
            avg_issue = sum(self._prefetch_issue_latencies) / n if self._prefetch_issue_latencies else 0.0
            stats = {
                "batches_prefetched": n,
                "avg_wait_ms": avg_wait * 1000.0,
                "avg_issue_to_wait_ms": avg_issue * 1000.0,
                "total_prefetched_ids": self._prefetch_total_ids,
                "embeddings_per_sec_during_wait": (self._prefetch_total_ids / sum(self._prefetch_wait_latencies)) if sum(self._prefetch_wait_latencies) > 0 else 0.0,
            }
        if reset:
            self._prefetch_issue_ts.clear()
            self._prefetch_sizes.clear()
            self._prefetch_wait_latencies.clear()
            self._prefetch_issue_latencies.clear()
            self._prefetch_total_ids = 0
        return stats

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(tables={self.feature_keys})"
