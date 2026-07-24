import os
from pathlib import Path

import torch


def load_ops_library() -> None:
    torch.ops.load_library(
        os.environ.get(
            "RECSTORE_OPS_LIBRARY",
            str(Path(__file__).resolve().parents[4] / "build/lib/lib_recstore_ops.so"),
        )
    )


if os.environ.get("RECSTORE_DEFER_OPS_LOAD") != "1":
    load_ops_library()

from .DistTensor import DistTensor
from .DistEmb import DistEmbedding
from .KVClient import RecStoreClient, get_kv_client
from .optimizer import SparseSGD
from . import bagpipe_cache
from torchrec_kv.EmbeddingBag import RecStoreEmbeddingBagCollection
# from .controller_process import (
#     KGCacheControllerWrapperBase,
#     KGCacheControllerWrapperDummy,
#     KGCacheControllerWrapper,
#     TestPerfSampler,
#     BasePerfSampler,
#     GraphCachedSampler
# )

# from .torch_op import *
