from itertools import product

import pytest
import torch
from torch_sparse.tensor import SparseTensor

from .utils import dtypes, devices, tensor


@pytest.mark.parametrize('dtype,device', product(dtypes, devices))
def test_cat(dtype, device):
    index = tensor([[0, 0, 1, 2], [0, 1, 2, 2]], torch.long, device)
    value = tensor([1, 2, 3, 4], dtype, device)
    mat = SparseTensor(index, value)
    mat.fill_cache_()

    mat = mat.remove_diag()
    index, value = mat.coo()
    assert index.tolist() == [[0, 1], [1, 2]]
    assert value.tolist() == [2, 3]
    assert len(mat.cached_keys()) == 2
    assert mat.storage.rowcount.tolist() == [1, 1, 0]
    assert mat.storage.colcount.tolist() == [0, 1, 1]
