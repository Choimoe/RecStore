import torch
from .KVClient import get_kv_client

class DistTensor:
    def __init__(self, shape: tuple, dtype: torch.dtype, name: str, init_func=None):
        if not isinstance(name, str) or not name:
            raise ValueError("DistTensor must have a valid name.")

        self._shape = shape
        self._dtype = dtype
        self._name = name
        self._kv_client = get_kv_client()

        if self._name not in self._kv_client.data_name_list():
            self._kv_client.init_data(self._name, self._shape, self._dtype, init_func)
        else:
            existing_dtype, existing_shape = self._kv_client.get_data_meta(self._name)
            if self._shape != existing_shape or self._dtype != existing_dtype:
                raise TypeError(
                    f"Tensor '{self._name}' already exists with a different shape or dtype."
                )

    def __getitem__(self, ids: torch.Tensor) -> torch.Tensor:
        if not isinstance(ids, torch.Tensor):
            ids = torch.tensor(ids, dtype=torch.int64)
        if ids.dtype != torch.int64:
            ids = ids.to(torch.int64)
        
        return self._kv_client.pull(self._name, ids)

    def __setitem__(self, ids: torch.Tensor, data: torch.Tensor):
        if not isinstance(ids, torch.Tensor):
            ids = torch.tensor(ids, dtype=torch.int64)
        if ids.dtype != torch.int64:
            ids = ids.to(torch.int64)
        
        # In the context of DistEmb, the 'data' passed to __setitem__ during
        # the backward pass will be gradients. We use the specialized 'update'
        # method for this. A more general client could inspect the data type
        # or have separate methods, but for now we assume __setitem__ is for updates.
        self._kv_client.update(self._name, ids, data)
    
    @property
    def shape(self) -> tuple:
        return self._shape

    @property
    def dtype(self) -> torch.dtype:
        return self._dtype

    @property
    def name(self) -> str:
        return self._name

    def __repr__(self) -> str:
        return f"DistTensor(name='{self.name}', shape={self.shape}, dtype={self.dtype})"
