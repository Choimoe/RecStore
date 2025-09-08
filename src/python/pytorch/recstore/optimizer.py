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
    """Handles the optimization step for a DistEmbedding module."""
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

    current_weights = mod.weight[unique_ids]
    updated_weights = current_weights - lr * summed_grads
    mod.weight[unique_ids] = updated_weights

def _process_generic_module_with_trace(mod: Any, lr: float, kv_client: Any):
    """Handles the optimization step for a generic module with _config_names and _trace."""
    if not mod._trace:
        return

    grads_by_table: Dict[str, List[Tuple[torch.Tensor, torch.Tensor]]] = {}
    for config_name, ids, grads in mod._trace:
        if config_name not in grads_by_table:
            grads_by_table[config_name] = []
        grads_by_table[config_name].append((ids, grads))

    for config_name, trace_list in grads_by_table.items():
        all_ids = torch.cat([ids for ids, _ in trace_list])
        all_grads = torch.cat([grads for _, grads in trace_list])

        unique_ids, inverse_indices = torch.unique(all_ids, return_inverse=True)

        embedding_dim = all_grads.shape[1]
        summed_grads = torch.zeros((len(unique_ids), embedding_dim), device=all_grads.device)
        summed_grads.index_add_(0, inverse_indices, all_grads)

        # Backend operations are on CPU
        unique_ids_cpu = unique_ids.cpu()
        summed_grads_cpu = summed_grads.cpu()

        current_weights = kv_client.pull(name=config_name, ids=unique_ids_cpu)
        updated_weights = current_weights - lr * summed_grads_cpu
        kv_client.push(name=config_name, ids=unique_ids_cpu, data=updated_weights)

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
                        _process_generic_module_with_trace(mod, lr, self.kv_client)
                    else:
                        print(f"Warning: Module type {type(mod).__name__} is not supported by SparseSGD optimizer.")
