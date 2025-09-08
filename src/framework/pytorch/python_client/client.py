import torch
import os
from typing import Optional

class RecstoreClient:
    _is_initialized = False

    def __init__(self, library_path: Optional[str] = None):
        if RecstoreClient._is_initialized:
            return
            
        if library_path is None:
            script_dir = os.path.dirname(__file__)
            default_lib_path = os.path.abspath(
                os.path.join(script_dir, '../../../../build/_recstore_ops.so')
            )
            if not os.path.exists(default_lib_path):
                 raise ImportError(
                    f"Could not find Recstore library at default path: {default_lib_path}\n"
                    "Please provide the correct path via the 'library_path' argument "
                    "or ensure your project is built correctly with build.sh."
                )
            library_path = default_lib_path

        torch.ops.load_library(library_path)
        self.ops = torch.ops.recstore_ops
        RecstoreClient._is_initialized = True

    def emb_read(self, keys: torch.Tensor, embedding_dim: int) -> torch.Tensor:
        if keys.dtype != torch.int64:
            raise TypeError(f"keys tensor must be of dtype torch.int64, but got {keys.dtype}")
        if not isinstance(embedding_dim, int) or embedding_dim <= 0:
            raise ValueError("embedding_dim must be a positive integer")

        return self.ops.emb_read(keys, embedding_dim)

    def emb_update(self, keys: torch.Tensor, grads: torch.Tensor) -> None:
        if keys.dtype != torch.int64:
            raise TypeError(f"keys tensor must be of dtype torch.int64, but got {keys.dtype}")
        if grads.dtype != torch.float32:
            raise TypeError(f"grads tensor must be of dtype torch.float32, but got {grads.dtype}")
        if keys.shape[0] != grads.shape[0]:
            raise ValueError("keys and grads must have the same number of entries")

        self.ops.emb_update(keys, grads)

    def emb_write(self, keys: torch.Tensor, values: torch.Tensor) -> None:
        if keys.dtype != torch.int64:
            raise TypeError(f"keys tensor must be of dtype torch.int64, but got {keys.dtype}")
        if values.dtype != torch.float32:
            raise TypeError(f"values tensor must be of dtype torch.float32, but got {values.dtype}")
        if keys.shape[0] != values.shape[0]:
            raise ValueError("keys and values must have the same number of entries")

        self.ops.emb_write(keys, values)
