import torch
from dgsparse.tensor import SparseTensor

# torch.ops.load_library("_spmm_cuda.so")

# torch.ops.dgsparse.SpMM


def spmm_sum(sparse: SparseTensor, dense: torch.Tensor,
             algorithm) -> torch.Tensor:
    r"""
    Matrix multiplication of a sparse tensor
    and a dense tensor with sum reduction.

    Args:
        sparse (SparseTensor): The sparse tensor.
        dense (Tensor): The dense tensor.

    rtype: :class:'Tensor'
    """
    has_value = sparse.has_value
    rowptr = sparse.storage._rowptr
    col = sparse.storage._col
    values = sparse.storage._values
    return torch.ops.dgsparse_spmm.spmm_sum(rowptr, col, values, dense,
                                            has_value, algorithm)


def spmm_mean(sparse: SparseTensor, dense: torch.Tensor,
              algorithm) -> torch.Tensor:
    r"""
    Matrix multiplication of a sparse tensor
    and a dense tensor with mean reduction.

    Args:
        sparse (SparseTensor): The sparse tensor.
        dense (Tensor): The dense tensor.

    rtype: :class:'Tensor'
    """
    has_value = sparse.has_value
    rowptr = sparse.storage._rowptr
    col = sparse.storage._col
    values = sparse.storage._values
    return torch.ops.dgsparse_spmm.spmm_mean(rowptr, col, values, dense,
                                             has_value, algorithm)


def spmm_max(sparse: SparseTensor, dense: torch.Tensor,
             algorithm) -> torch.Tensor:
    r"""
    Matrix multiplication-like of a sparse tensor
    and a dense tensor with mean reduction.

    Args:
        sparse (SparseTensor): The sparse tensor.
        dense (Tensor): The dense tensor.

    rtype: :class:'Tensor'
    """
    has_value = sparse.has_value
    rowptr = sparse.storage._rowptr
    col = sparse.storage._col
    values = sparse.storage._values
    return torch.ops.dgsparse_spmm.spmm_max(rowptr, col, values, dense,
                                            has_value, algorithm)


def spmm_min(sparse: SparseTensor, dense: torch.Tensor,
             algorithm) -> torch.Tensor:
    r"""
    Matrix multiplication-like of a sparse tensor
    and a dense tensor with mean reduction.

    Args:
        sparse (SparseTensor): The sparse tensor.
        dense (Tensor): The dense tensor.

    rtype: :class:'Tensor'
    """
    has_value = sparse.has_value
    rowptr = sparse.storage._rowptr
    col = sparse.storage._col
    values = sparse.storage._values
    return torch.ops.dgsparse_spmm.spmm_min(rowptr, col, values, dense,
                                            has_value, algorithm)
