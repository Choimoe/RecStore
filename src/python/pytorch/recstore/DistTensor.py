import torch
from .KVClient import get_kv_client
from typing import Optional, Tuple, Callable, Any


def _default_init_data(shape, dtype):
    """Default UDF for data initialization."""
    return torch.zeros(shape, dtype=dtype)

class DistTensor:
    def __init__(
        self,
        shape: Tuple[int, ...],
        dtype: torch.dtype,
        name: Optional[str] = None,
        init_func: Optional[Callable] = None,
        part_policy: Any = None,
        persistent: bool = False,
        is_gdata: bool = True,
    ):
        if not isinstance(name, str) or not name:
            raise ValueError("DistTensor must have a valid name.")

        self._shape = shape
        self._dtype = dtype
        self._name = name
        self._kv_client = get_kv_client()
        self._is_gdata = is_gdata
        self._persistent = persistent

        if part_policy is not None:
            print("Warning: part_policy is ignored in the current ops-based implementation.")

        self._part_policy = part_policy

        if init_func is None:
            init_func = _default_init_data

        exist_names = self._kv_client.data_name_list()

        if name is None:
            assert not persistent, "We cannot generate anonymous persistent distributed tensors."
            global DIST_TENSOR_ID
            name = f"anonymous-tensor-{DIST_TENSOR_ID}"
            DIST_TENSOR_ID += 1

        if self._name not in self._kv_client.data_name_list():
            self._owner = True
            self._kv_client.init_data(
                self._name, self._shape, self._dtype, part_policy, init_func, is_gdata
            )
        else:
            self._owner = False
            existing_dtype, existing_shape = self._kv_client.get_data_meta(self._name)
            if self._shape != existing_shape or self._dtype != existing_dtype:
                raise TypeError(
                    f"Tensor '{self._name}' already exists with a different shape or dtype. "
                    f"Existing: {existing_shape}/{existing_dtype}, "
                    f"New: {self._shape}/{self._dtype}"
                )

    def __del__(self):
        """Destructor to clean up non-persistent tensors."""
        try:
            if not self._persistent and self._owner:
                self._kv_client.delete_data(self._name)
        except Exception as e:
            print(f"Warning: Failed to delete DistTensor '{self._name}'. Reason: {e}")
            pass

    def __getitem__(self, ids: Any) -> torch.Tensor:
        """Pulls data corresponding to the given IDs."""
        if not isinstance(ids, torch.Tensor):
            ids = torch.tensor(ids)
        if ids.dtype != torch.int64:
            ids = ids.to(torch.int64)
        
        return self._kv_client.pull(self._name, ids)

    def __setitem__(self, ids: Any, data: torch.Tensor):
        """Pushes data to the given IDs."""
        if not isinstance(ids, torch.Tensor):
            ids = torch.tensor(ids)
        if ids.dtype != torch.int64:
            ids = ids.to(torch.int64)
        
        # In the context of DistEmb, the 'data' passed to __setitem__ during
        # the backward pass will be gradients. We use the specialized 'update'
        # method for this. A more general client could inspect the data type
        # or have separate methods, but for now we assume __setitem__ is for updates.
        self._kv_client.update(self._name, ids, data)
    
    def __len__(self) -> int:
        """Returns the size of the first dimension."""
        return self._shape[0]

    @property
    def shape(self) -> Tuple[int, ...]:
        """Return the shape of the distributed tensor."""
        return self._shape

    @property
    def dtype(self) -> torch.dtype:
        """Return the data type of the distributed tensor."""
        return self._dtype

    @property
    def name(self) -> str:
        """Return the name of the distributed tensor."""
        return self._name

    @property
    def tensor_name(self) -> str:
        """Return the tensor name."""
        return self._tensor_name

    def count_nonzero(self) -> int:
        """Count and return the number of nonzero value."""
        raise NotImplementedError("count_nonzero is not implemented for the ops-based client.")


    def __repr__(self) -> str:
        return f"DistTensor(name='{self.name}', shape={self.shape}, dtype={self.dtype})"
