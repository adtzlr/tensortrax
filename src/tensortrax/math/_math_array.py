"""
tensorTRAX: Math on (Hyper-Dual) Tensors with Trailing Axes.
"""
import numpy as np

from .._tensor import Tensor, f


def eye(A):
    "Identity (Eye) of a Tensor."
    if isinstance(A, Tensor):
        B = np.zeros_like(f(A))
    else:
        B = np.zeros_like(A)
    B[np.diag_indices(B.shape[0])] = 1
    return B


def cross(a, b):
    "Cross product of two vectors a and b."
    return np.einsum(
        "...i->i...", np.cross(np.einsum("i...->...i", a), np.einsum("i...->...i", b))
    )
