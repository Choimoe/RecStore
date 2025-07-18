import torch
from .DistTensor import DistTensor
from typing import Optional, Callable, Any

class _DistEmbFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, ids, dist_tensor, dummy_param):
        ctx.save_for_backward(ids)
        ctx.dist_tensor = dist_tensor
        embs = dist_tensor[ids]
        return embs

    @staticmethod
    def backward(ctx, grad_output):
        ids, = ctx.saved_tensors
        dist_tensor = ctx.dist_tensor
        dist_tensor[ids] = grad_output.contiguous()
        return None, None, None

class DistEmbedding(torch.nn.Module):
    """
    Distributed node embeddings.

    This class, inspired by DGL's DistEmbedding, provides a way to handle large-scale
    learnable embeddings for models. It uses a DistTensor backend for storage and
    retrieval.

    Updates to these embeddings are handled sparsely. Instead of using PyTorch's
    standard autograd for the backward pass, this module traces the forward
    tensors. A custom optimizer is then required to process these traces and apply gradients.

    Parameters
    ----------
    num_embeddings : int
        The total number of embeddings in the layer.
    embedding_dim : int
        The dimensionality of each embedding vector.
    name : str, optional
        A unique name for the embeddings. If not provided, a name will be
        generated, but the embedding will not be persistent.
    init_func : callable, optional
        A function to initialize the embedding weights. If None, they are
        initialized to zeros.
    part_policy : Any, optional
        The partition policy for distributing the embeddings. This is currently
        a placeholder for API compatibility and is not used by the ops-based backend.
    """
    def __init__(
        self,
        num_embeddings: int,
        embedding_dim: int,
        name: Optional[str] = None,
        init_func: Optional[Callable] = None,
        part_policy: Any = None,
        persistent: bool = False
    ):
        super(DistEmbedding, self).__init__()
        if not name:
            raise ValueError("DistEmb requires a unique 'name'.")
        
        self._tensor = DistTensor(
            shape=(num_embeddings, embedding_dim),
            dtype=torch.float32,
            name=name,
            init_func=init_func,
            part_policy=part_policy,
            persistent=persistent
        )
        self.dummy_param = torch.nn.Parameter(torch.empty(0))

        self._trace = []
        self._num_embeddings = num_embeddings
        self._embedding_dim = embedding_dim
        self._part_policy = part_policy
        self._optm_state = None
        self.persistent = persistent

    def forward(self, ids):
        """
        Performs a lookup for the given embedding IDs.

        If called within a gradient-enabled context (e.g., model.train()),
        it traces the pulled embeddings so that a custom optimizer can
        later apply sparse gradients.

        Parameters
        ----------
        ids : torch.Tensor
            A tensor of IDs to look up in the embedding table.

        Returns
        -------
        torch.Tensor
            The corresponding embeddings for the given IDs.
        """
        return _DistEmbFunction.apply(ids, self._tensor, self.dummy_param)

    def reset_trace(self):
        """Reset the traced data. Should be called after each optimizer step."""
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

    @property
    def part_policy(self) -> Any:
        """Return the partition policy."""
        return self._part_policy

    def __repr__(self):
        return (f"DistEmbedding(name='{self.name}', num_embeddings={self.num_embeddings}, "
                f"embedding_dim={self.embedding_dim})")
