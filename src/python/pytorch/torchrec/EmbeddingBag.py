import torch
import logging
from torch.autograd import Function
from typing import List, Dict, Any, Tuple
from torchrec.sparse.jagged_tensor import KeyedJaggedTensor, KeyedTensor

from ..recstore.KVClient import get_kv_client, RecStoreClient

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class _RecStoreEBCFunction(Function):
    """
    Custom autograd Function to bridge PyTorch's autograd engine with the
    RecStore backend. It handles the forward lookup and backward gradient update.
    """

    @staticmethod
    def forward(
        ctx,
        features: KeyedJaggedTensor,
        module: 'RecStoreEmbeddingBagCollection'
    ) -> KeyedJaggedTensor:
        """
        Performs the forward pass by pulling embedding vectors from RecStore.

        Args:
            ctx: PyTorch autograd context to save tensors for the backward pass.
            features (KeyedJaggedTensor): Input features containing IDs to look up.
            module (RecStoreEmbeddingBagCollection): The parent module instance to
                access the kv_client.

        Returns:
            KeyedJaggedTensor: The resulting embedding vectors, structured as a KJT.
        """
        ctx.features = features
        ctx.module = module
        
        grouped_features = features.to_dict()
        pulled_embs: Dict[str, torch.Tensor] = {}

        for key in features.keys():
            ids_to_pull = grouped_features[key].values()
            if ids_to_pull.numel() == 0:
                # Create an empty tensor with the correct embedding dimension if no IDs are present
                emb_dim = module._embedding_dims[key]
                pulled_embs[key] = torch.empty(
                    (0, emb_dim), dtype=torch.float32, device=features.device()
                )
                continue

            pulled_embs[key] = module.kv_client.pull(name=key, ids=ids_to_pull)

        # Reconstruct the KeyedJaggedTensor from the pulled embeddings
        output_values = torch.cat([pulled_embs[key] for key in features.keys()], dim=0)
        output_kt = KeyedTensor(
            keys=features.keys(),
            values=output_values,
            lengths=features.lengths(),
        )
        
        return KeyedJaggedTensor.from_keyed_tensor(kt=output_kt)

    @staticmethod
    def backward(ctx, grad_output: KeyedJaggedTensor) -> Tuple[None, None]:
        """
        Performs the backward pass by pushing gradients to the RecStore backend.

        Args:
            ctx: PyTorch autograd context with saved tensors from the forward pass.
            grad_output (KeyedJaggedTensor): Gradients from the upstream layers.

        Returns:
            A tuple of None values, as gradients are handled manually and not
            propagated further back for the inputs of this function.
        """
        features: KeyedJaggedTensor = ctx.features
        module: 'RecStoreEmbeddingBagCollection' = ctx.module
        
        grouped_features = features.to_dict()
        grouped_grads = grad_output.to_dict()

        for key in features.keys():
            ids_to_update = grouped_features[key].values()
            if ids_to_update.numel() == 0:
                continue
            
            grads = grouped_grads[key].values()
            module.kv_client.update(name=key, ids=ids_to_update, grads=grads.contiguous())

        return None, None


class RecStoreEmbeddingBagCollection(torch.nn.Module):
    """
    An EmbeddingBagCollection that uses a custom RecStore backend for storage
    and computation, designed to be a drop-in replacement for torchrec's EBC.
    """
    def __init__(self, embedding_bag_configs: List[Dict[str, Any]]):
        """
        Initializes the RecStoreEmbeddingBagCollection.

        Args:
            embedding_bag_configs (List[Dict[str, Any]]): A list of configs,
                where each config is a dict describing an embedding table.
                Required keys: 'name', 'num_embeddings', 'embedding_dim'.
        """
        super().__init__()
        
        if not embedding_bag_configs:
            raise ValueError("embedding_bag_configs cannot be empty.")

        self._embedding_bag_configs = embedding_bag_configs
        self.kv_client: RecStoreClient = get_kv_client()
        self._embedding_dims: Dict[str, int] = {}

        logging.info("Initializing RecStoreEmbeddingBagCollection...")
        for config in self._embedding_bag_configs:
            self._validate_config(config)
            name: str = config["name"]
            num_embeddings: int = config["num_embeddings"]
            embedding_dim: int = config["embedding_dim"]
            
            self._embedding_dims[name] = embedding_dim
            
            logging.info(
                f"  - Initializing table '{name}' in RecStore backend with "
                f"shape ({num_embeddings}, {embedding_dim})."
            )
            
            self.kv_client.init_data(
                name=name,
                shape=(num_embeddings, embedding_dim),
                dtype=torch.float32,
                init_func=None
            )
        logging.info("RecStoreEmbeddingBagCollection initialized successfully.")

    def _validate_config(self, config: Dict[str, Any]):
        """Helper to validate a single embedding table configuration."""
        required_keys = ["name", "num_embeddings", "embedding_dim"]
        for key in required_keys:
            if key not in config:
                raise ValueError(f"Missing required key '{key}' in embedding_bag_configs.")

    def forward(self, features: KeyedJaggedTensor) -> KeyedJaggedTensor:
        """
        Performs the embedding lookup.

        The actual lookup and gradient handling logic is delegated to the
        _RecStoreEBCFunction to integrate with PyTorch's autograd.

        Args:
            features (KeyedJaggedTensor): The input features from the DLRM model.

        Returns:
            KeyedJaggedTensor: The looked-up embedding vectors.
        """
        return _RecStoreEBCFunction.apply(features, self)

    def __repr__(self) -> str:
        tables = list(self._embedding_dims.keys())
        return (
            f"{self.__class__.__name__}(\n"
            f"  (tables): {tables}\n"
            f")"
        )
