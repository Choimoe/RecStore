import torch
import logging
from torch.autograd import Function
from typing import List, Dict, Any, Tuple
from torchrec.sparse.jagged_tensor import KeyedJaggedTensor, KeyedTensor
from torchrec.modules.embedding_configs import EmbeddingBagConfig
from ..recstore.KVClient import get_kv_client, RecStoreClient

# Configure logging for better diagnostics
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)


class _RecStoreEBCFunction(Function):
    """
    FX-traceable autograd Function to bridge PyTorch's autograd engine with
    the RecStore backend.
    """

    @staticmethod
    def forward(
        ctx,
        # Non-tensor arguments are passed first
        module: "RecStoreEmbeddingBagCollection",
        feature_keys: List[str],
        # Tensor arguments (which become Proxies during tracing) are passed next
        features_values: torch.Tensor,
        features_lengths: torch.Tensor,
    ) -> torch.Tensor:
        """
        Performs the forward pass by pulling embedding vectors from RecStore.
        This implementation is designed to be torch.fx.traceable.
        """
        # Reconstruct a KJT from the raw tensors for ID lookup.
        # This KJT is temporary and not used in a way that breaks tracing.
        features = KeyedJaggedTensor(
            keys=feature_keys,
            values=features_values,
            lengths=features_lengths,
        )
        
        ctx.save_for_backward(features_values, features_lengths)
        ctx.module = module
        ctx.feature_keys = feature_keys

        pulled_embs: List[torch.Tensor] = []
        lengths = features.lengths()
        values = features.values()
        # Use static index-based access to avoid iterating over Proxy
        start = 0
        for i in range(len(feature_keys)):
            l = lengths[i]
            ids_to_pull = values[start:start+l]
            pulled_embs.append(module.kv_client.pull(name=feature_keys[i], ids=ids_to_pull))
            start = start + l
        return torch.cat(pulled_embs, dim=0)

    @staticmethod
    def backward(
        ctx, grad_output_values: torch.Tensor
    ) -> Tuple[None, None, None, None]:
        """
        Performs the backward pass by pushing gradients to the RecStore backend.
        This implementation is also FX-traceable.
        """
        features_values, features_lengths = ctx.saved_tensors
        module: "RecStoreEmbeddingBagCollection" = ctx.module
        feature_keys: List[str] = ctx.feature_keys

        # Reconstruct the original KJT to get IDs and the gradient KT to get gradients.
        features = KeyedJaggedTensor(
            keys=feature_keys,
            values=features_values,
            lengths=features_lengths,
        )
        grad_output = KeyedTensor(
            keys=feature_keys,
            values=grad_output_values,
            lengths=features_lengths, # Grads have the same length structure as inputs
        )

        lengths = features.lengths()
        values = features.values()
        grad_values = grad_output.values()
        start = 0
        grad_start = 0
        for i in range(len(feature_keys)):
            l = lengths[i]
            ids_to_update = values[start:start+l]
            grads = grad_values[grad_start:grad_start+l]
            module.kv_client.update(name=feature_keys[i], ids=ids_to_update, grads=grads.contiguous())
            start = start + l
            grad_start = grad_start + l

        # Return gradients for inputs to forward: module, feature_keys, features_values, features_lengths
        return None, None, None, None


class RecStoreEmbeddingBagCollection(torch.nn.Module):
    """
    An FX-traceable EmbeddingBagCollection that uses a custom RecStore backend.
    It is designed as a drop-in replacement for torchrec.EmbeddingBagCollection
    within a DLRM model.
    """

    def __init__(self, embedding_bag_configs: List[Dict[str, Any]]):
        super().__init__()

        if not embedding_bag_configs:
            raise ValueError("embedding_bag_configs cannot be empty.")

        # Convert dicts to EmbeddingBagConfig objects
        self._embedding_bag_configs = [
            EmbeddingBagConfig(
                name=c["name"],
                embedding_dim=c["embedding_dim"],
                num_embeddings=c["num_embeddings"],
                feature_names=c.get("feature_names", [c["name"]])
            )
            for c in embedding_bag_configs
        ]
        self.kv_client: RecStoreClient = get_kv_client()
        
        # Store feature names and embedding dimensions in a static, ordered way.
        # This is crucial for FX tracing, as we will iterate over these static lists.
        self.feature_keys: List[str] = [c.name for c in self._embedding_bag_configs]
        self._embedding_dims: Dict[str, int] = {c.name: c.embedding_dim for c in self._embedding_bag_configs}

        logging.info("Initializing RecStoreEmbeddingBagCollection...")
        for config in self._embedding_bag_configs:
            self._validate_config(config)
            name: str = config.name
            num_embeddings: int = config.num_embeddings
            embedding_dim: int = config.embedding_dim

            logging.info(
                f"  - Ensuring table '{name}' exists in RecStore backend with "
                f"shape ({num_embeddings}, {embedding_dim})."
            )
            self.kv_client.init_data(
                name=name,
                shape=(num_embeddings, embedding_dim),
                dtype=torch.float32,
            )
        logging.info("RecStoreEmbeddingBagCollection initialized successfully.")

    def embedding_bag_configs(self):
        return self._embedding_bag_configs

    def _validate_config(self, config: EmbeddingBagConfig):
        """Helper to validate a single embedding table configuration."""
        required_keys = ["name", "num_embeddings", "embedding_dim"]
        for key in required_keys:
            if not hasattr(config, key):
                raise ValueError(
                    f"Missing required key '{key}' in embedding_bag_configs."
                )

    def forward(self, features: KeyedJaggedTensor) -> KeyedTensor:
        """
        Performs the embedding lookup in a way that is compatible with torch.fx.
        """
        # Decompose the KJT into its constituent tensors before passing to the
        # autograd.Function. This is a key pattern for making complex objects
        # traceable, as fx traces operations on tensors.
        pooled_embs_values = _RecStoreEBCFunction.apply(
            self,
            self.feature_keys,
            features.values(),
            features.lengths(),
        )

        # Reconstruct the final KeyedTensor, which is the expected output format
        # for the sparse architecture of the DLRM.
        # Only pass supported arguments to KeyedTensor
        return KeyedTensor(
            keys=self.feature_keys,
            values=pooled_embs_values,
            length_per_key=features.lengths(),
        )

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(tables={self.feature_keys})"
