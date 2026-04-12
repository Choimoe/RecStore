import torch
from typing import List, Union, Dict, Tuple, Any

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
                        self._inflight_handles.extend(
                            _process_generic_module_with_trace(mod, lr, self.kv_client)
                        )
                    else:
                        print(f"Warning: Module type {type(mod).__name__} is not supported by SparseSGD optimizer.")
                    if hasattr(mod, 'reset_trace'):
                        mod.reset_trace()
