import torch


def load_recstore_library():
    torch.ops.load_library(
        "../../../build/lib/librecstore_pytorch.so")
    torch.classes.load_library(
        "../../../build/lib/librecstore_pytorch.so")


load_recstore_library()

merge_op = torch.ops.librecstore_pytorch.merge_op
uva_cache_query_op = torch.ops.librecstore_pytorch.uva_cache_query_op
init_folly = torch.ops.librecstore_pytorch.init_folly
construct_renumbering_dict_op = torch.ops.librecstore_pytorch.construct_renumbering_dict_op
NarrowShapeTensor_op = torch.ops.librecstore_pytorch.NarrowShapeTensor

GpuCache = torch.classes.librecstore_pytorch.GpuCache
IPCTensorFactory = torch.classes.librecstore_pytorch.IPCTensorFactory
KGCacheController = torch.classes.librecstore_pytorch.KGCacheController
ZipfianTorchFiller = torch.classes.librecstore_pytorch.ZipfianTorchFiller

MultiProcessBarrierFactory = torch.classes.librecstore_pytorch.MultiProcessBarrierFactory

Mfence = torch.classes.librecstore_pytorch.Mfence
