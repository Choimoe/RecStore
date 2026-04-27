import torch
from typing import List, Union, Dict, Tuple, Any
from .single_node_exchange import SparseGradPayload, exchange_sparse_grads

_LOCAL_FAST_PATH_BACKENDS = {"local_shm", "hierkv"}

class DistEmbedding:
    pass

def _get_kv_client_if_needed(params: List[Any]):
    """Dynamically imports and returns the KV client if params are provided."""
    if params:
        from .KVClient import get_kv_client
        from .DistEmb import DistEmbedding as DistEmbeddingImpl
        global DistEmbedding
        DistEmbedding = DistEmbeddingImpl
        return get_kv_client()
    return None

def _process_dist_embedding_module(mod: DistEmbedding, lr: float):
    """Handles the optimization step for a DistEmbedding module using gradient accumulation."""
    if not mod._trace:
        return

    all_ids = torch.cat([ids for ids, _ in mod._trace])
    all_grads = torch.cat([grads for _, grads in mod._trace])

    unique_ids, inverse_indices = torch.unique(all_ids, return_inverse=True)

    summed_grads = torch.zeros(
        (len(unique_ids), mod.embedding_dim),
        device=all_grads.device,
        dtype=all_grads.dtype
    )
    summed_grads.index_add_(0, inverse_indices, all_grads)

    scaled_grads = lr * summed_grads
    
    current_weights = mod.weight[unique_ids]
    updated_weights = current_weights - scaled_grads
    mod.weight[unique_ids] = updated_weights

def _process_generic_module_with_trace(mod: Any, lr: float, kv_client: Any):
    """Handles sparse trace aggregation and backend updates for generic modules."""
    if not mod._trace:
        return []

    traces_by_name: Dict[str, List[Tuple[torch.Tensor, torch.Tensor]]] = {}
    for entry in mod._trace:
        if isinstance(entry, dict):
            name = entry["name"]
            ids = entry["ids"]
            if "grads" in entry:
                grads = entry["grads"]
            else:
                grads = entry["grad"].unsqueeze(0).expand(int(entry["count"]), -1)
        else:
            name, ids, grads = entry
        traces_by_name.setdefault(name, []).append((ids, grads))

    module_kv_client = getattr(mod, "kv_client", None) or kv_client
    handles = []
    for name, entries in traces_by_name.items():
        all_ids = torch.cat([ids for ids, _ in entries], dim=0)
        all_grads = torch.cat([grads for _, grads in entries], dim=0)

        unique_ids, inverse_indices = torch.unique(all_ids, return_inverse=True)
        summed_grads = torch.zeros(
            (len(unique_ids), all_grads.size(1)),
            device=all_grads.device,
            dtype=all_grads.dtype,
        )
        summed_grads.index_add_(0, inverse_indices, all_grads)

        # Backend sparse optimizers own learning-rate application for these modules.
        handles.append(
            (
                module_kv_client,
                module_kv_client.update_async(name=name, ids=unique_ids, grads=summed_grads),
            )
        )
    return handles


def _can_use_single_node_distributed_fast_path(mod: Any) -> bool:
    if not getattr(mod, "enable_single_node_distributed_fast_path", False):
        return False
    if getattr(mod, "single_node_distributed_mode", None) != "single_node":
        return False
    if getattr(mod, "single_node_owner_policy", None) != "hash_mod_world_size":
        return False
    if getattr(mod, "single_node_ps_backend", None) not in _LOCAL_FAST_PATH_BACKENDS:
        return False
    dist = getattr(torch, "distributed", None)
    if dist is None or not hasattr(dist, "is_initialized") or not dist.is_initialized():
        return False
    if not hasattr(dist, "get_world_size") or dist.get_world_size() <= 1:
        return False
    return True


def _collect_traces_by_name(mod: Any) -> Dict[str, List[Tuple[torch.Tensor, torch.Tensor]]]:
    traces_by_name: Dict[str, List[Tuple[torch.Tensor, torch.Tensor]]] = {}
    for entry in mod._trace:
        if isinstance(entry, dict):
            name = entry["name"]
            ids = entry["ids"]
            if "grads" in entry:
                grads = entry["grads"]
            else:
                grads = entry["grad"].unsqueeze(0).expand(int(entry["count"]), -1)
        else:
            name, ids, grads = entry
        traces_by_name.setdefault(name, []).append((ids, grads))
    return traces_by_name


def _aggregate_ids_and_grads(
    ids: torch.Tensor,
    grads: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    unique_ids, inverse_indices = torch.unique(ids, return_inverse=True)
    summed_grads = torch.zeros(
        (len(unique_ids), grads.size(1)),
        device=grads.device,
        dtype=grads.dtype,
    )
    summed_grads.index_add_(0, inverse_indices, grads)
    return unique_ids, summed_grads


def _process_generic_module_with_trace_single_node_distributed(mod: Any) -> None:
    if not mod._trace:
        return

    dist = torch.distributed
    rank = int(dist.get_rank())
    world_size = int(dist.get_world_size())
    exchange_backend = dist
    module_kv_client = getattr(mod, "kv_client", None)
    if module_kv_client is None:
        raise RuntimeError("single-node distributed sparse update requires module kv_client")

    traces_by_name = _collect_traces_by_name(mod)
    for name, entries in traces_by_name.items():
        all_ids = torch.cat([ids for ids, _ in entries], dim=0)
        all_grads = torch.cat([grads for _, grads in entries], dim=0)
        unique_ids, summed_grads = _aggregate_ids_and_grads(all_ids, all_grads)
        if getattr(mod, "single_node_owner_policy", "hash_mod_world_size") != "hash_mod_world_size":
            raise RuntimeError("single-node distributed sparse update currently requires hash_mod_world_size")

        normalized_ids = unique_ids.detach().to(dtype=torch.int64)
        normalized_grads = summed_grads.detach().to(dtype=torch.float32)
        destination_ranks = torch.remainder(normalized_ids, world_size)
        payload_device = normalized_ids.device

        local_payload = SparseGradPayload(
            rank=rank,
            destination_ranks=destination_ranks,
            source_ranks=torch.full(
                (normalized_ids.numel(),),
                rank,
                dtype=torch.int64,
                device=payload_device,
            ),
            row_positions=torch.arange(
                normalized_ids.numel(),
                dtype=torch.int64,
                device=payload_device,
            ),
            fused_ids=normalized_ids,
            grads=normalized_grads,
        )
        gathered_payloads = exchange_sparse_grads(
            local_payload,
            world_size=world_size,
            backend=exchange_backend,
        )

        owner_ids: list[int] = []
        owner_grads: list[torch.Tensor] = []
        for payload in gathered_payloads:
            owner_ids.extend(payload.fused_ids.detach().cpu().tolist())
            owner_grads.append(payload.grads)

        if not owner_ids:
            continue

        owner_grads_tensor = torch.cat(owner_grads, dim=0)
        owner_ids_tensor = torch.tensor(
            owner_ids,
            dtype=torch.int64,
            device=owner_grads_tensor.device,
        )
        owner_unique_ids, owner_summed_grads = _aggregate_ids_and_grads(
            owner_ids_tensor,
            owner_grads_tensor,
        )
        module_kv_client.local_update_flat(
            name=name,
            ids=owner_unique_ids,
            grads=owner_summed_grads,
        )

# --- Core Classes ---

class SparseOptimizer:
    """
    Base class for sparse optimizers.
    It handles updating parameters of modules like DistEmbedding.
    """
    def __init__(self, params: List[Union[DistEmbedding, torch.nn.Module]], lr: float):
        """
        Initializes the optimizer.

        Parameters
        ----------
        params : List[Union[DistEmbedding, torch.nn.Module]]
            A list of modules to be optimized.
        lr : float
            The learning rate.
        """
        self.param_groups = [{"params": params, "lr": lr}]
        self.kv_client = _get_kv_client_if_needed(params)
        self._inflight_handles: List[Tuple[Any, int]] = []

    def step(self):
        """
        Performs a single optimization step.
        This method should be implemented by subclasses.
        """
        raise NotImplementedError("The step() method must be implemented by a subclass.")

    def zero_grad(self):
        """
        Clears the traces of all parameter groups.
        """
        for group in self.param_groups:
            for mod in group["params"]:
                if hasattr(mod, 'reset_trace'):
                    mod.reset_trace()
                # else:
                #     if hasattr(mod, 'grad') and mod.grad is not None:
                #         mod.grad.detach_()
                #         mod.grad.zero_()

    def flush(self):
        """Wait for all in-flight async sparse updates."""
        if self.kv_client is None:
            self._inflight_handles.clear()
            return
        for kv_client, handle in self._inflight_handles:
            kv_client.wait(handle)
        self._inflight_handles.clear()

class SparseSGD(SparseOptimizer):
    def step(self):
        """Performs a single Sparse SGD optimization step."""
        with torch.no_grad():
            for group in self.param_groups:
                lr = group["lr"]
                for mod in group["params"]:
                    if isinstance(mod, DistEmbedding):
                        _process_dist_embedding_module(mod, lr)
                    elif hasattr(mod, '_config_names') and hasattr(mod, '_trace'):
                        if _can_use_single_node_distributed_fast_path(mod):
                            _process_generic_module_with_trace_single_node_distributed(mod)
                        else:
                            self._inflight_handles.extend(
                                _process_generic_module_with_trace(mod, lr, self.kv_client)
                            )
                    else:
                        print(f"Warning: Module type {type(mod).__name__} is not supported by SparseSGD optimizer.")
                    if hasattr(mod, 'reset_trace'):
                        mod.reset_trace()
