import torch
from .DistTensor import DistTensor
from typing import Optional, Callable, Any

class _DistEmbFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, ids, dist_tensor, dummy_param, module_instance):
        """
        Forward pass for distributed embedding lookup.

        It saves the module instance to the context so the backward pass can
        access its trace list.
        """
        ctx.save_for_backward(ids)
        ctx.module_instance = module_instance
        embs = dist_tensor[ids]
        return embs

    @staticmethod
    def backward(ctx, grad_output):
        """
        Backward pass for distributed embedding.

        Instead of performing an update, this function now appends the
        (ids, gradients) pair to the module's trace list. An external
        optimizer is responsible for processing this trace.
        """
        ids, = ctx.saved_tensors
        module_instance = ctx.module_instance
        
        module_instance._trace.append((ids.detach(), grad_output.detach()))

        return None, None, None, None

class DistEmbedding(torch.nn.Module):
    """
    Distributed node embeddings.

    This module handles large-scale embeddings using a DistTensor backend.
    
    Instead of performing immediate gradient updates in the backward pass,
    it traces the IDs and gradients. A dedicated `SparseOptimizer` must be
    used to process this trace and apply the updates after `loss.backward()`
    is called.

    Parameters
    ----------
    num_embeddings : int
        The total number of embeddings.
    embedding_dim : int
        The dimensionality of each embedding vector.
    name : str
        A unique name for the embedding table.
    init_func : callable, optional
        A function to initialize the embedding weights. Defaults to zeros.
    """
    def __init__(
        self,
        num_embeddings: int,
        embedding_dim: int,
        name: str,
        init_func: Optional[Callable] = None,
        # part_policy is kept for API compatibility but not used in this backend
        part_policy: Any = None, 
    ):
        super(DistEmbedding, self).__init__()
        if not name:
            raise ValueError("DistEmbedding requires a unique 'name'.")
        
        self._tensor = DistTensor(
            shape=(num_embeddings, embedding_dim),
            dtype=torch.float32,
            name=name,
            init_func=init_func,
        )
        # A dummy parameter to ensure this module is included in the autograd graph
        self.dummy_param = torch.nn.Parameter(torch.empty(0))

        self._trace = []
        self._num_embeddings = num_embeddings
        self._embedding_dim = embedding_dim

    def forward(self, ids):
        """
        Performs a lookup for the given embedding IDs.

        If `torch.is_grad_enabled()` is true, it traces the lookup so that a
        sparse optimizer can later apply gradients.
        """
        return _DistEmbFunction.apply(ids, self._tensor, self.dummy_param, self)

    def reset_trace(self):
        """Reset the traced data. Should be called by the optimizer after a step."""
        self._trace = []

    @property
    def name(self) -> str:
        """Return the name of the embeddings."""
        return self._tensor.name

    @property
    def num_embeddings(self) -> int:
        """Return the number of embeddings."""
        return self._num_embeddings

    @property
    def embedding_dim(self) -> int:
        """Return the dimension of embeddings."""
        return self._embedding_dim

    @property
    def weight(self) -> DistTensor:
        """Return the DistTensor that stores the embeddings."""
        return self._tensor

    def __repr__(self):
        return (f"DistEmbedding(name='{self.name}', num_embeddings={self.num_embeddings}, "
                f"embedding_dim={self.embedding_dim})")
