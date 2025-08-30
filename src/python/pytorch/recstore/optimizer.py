import torch
from typing import List, Union

class DistEmbedding:
    pass

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
            A list of modules to be optimized. These modules are expected
            to have a `_trace` attribute and a `reset_trace` method.
        lr : float
            The learning rate.
        """
        self.param_groups = [{"params": params, "lr": lr}]
        self.kv_client = None
        # Dynamically import get_kv_client to avoid potential import cycles
        if params:
            from .KVClient import get_kv_client
            from .DistEmb import DistEmbedding as DistEmbeddingImpl
            global DistEmbedding
            DistEmbedding = DistEmbeddingImpl
            self.kv_client = get_kv_client()

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
                else:
                    if hasattr(mod, 'grad') and mod.grad is not None:
                        mod.grad.detach_()
                        mod.grad.zero_()

class SparseSGD(SparseOptimizer):
    def step(self):
        with torch.no_grad():
            for group in self.param_groups:
                lr = group["lr"]
                for mod in group["params"]:
                    if isinstance(mod, DistEmbedding):
                        if not mod._trace:
                            continue
                        
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

                    elif hasattr(mod, '_config_names') and hasattr(mod, '_trace'):
                        if not mod._trace:
                            continue
                        
                        for config_name, ids, grads in mod._trace:
                            unique_ids, inverse_indices = torch.unique(ids, return_inverse=True)

                            summed_grads = torch.zeros((len(unique_ids), grads.shape[1]), device=grads.device)
                            summed_grads.index_add_(0, inverse_indices, grads)

                            current_weights = self.kv_client.pull(name=config_name, ids=unique_ids)
                            updated_weights = current_weights - lr * summed_grads
                            self.kv_client.push(name=config_name, ids=unique_ids, data=updated_weights)
                    
                    else:
                        print(f"Warning: Module type {type(mod).__name__} is not supported by SparseSGD optimizer.")

