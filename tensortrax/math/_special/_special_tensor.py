r"""
 _                            
| |                          ████████╗██████╗  █████╗ ██╗  ██╗
| |_ ___ _ __  ___  ___  _ __╚══██╔══╝██╔══██╗██╔══██╗╚██╗██╔╝
| __/ _ \ '_ \/ __|/ _ \| '__|  ██║   ██████╔╝███████║ ╚███╔╝ 
| ||  __/ | | \__ \ (_) | |     ██║   ██╔══██╗██╔══██║ ██╔██╗ 
 \__\___|_| |_|___/\___/|_|     ██║   ██║  ██║██║  ██║██╔╝ ██╗
                                ╚═╝   ╚═╝  ╚═╝╚═╝  ╚═╝╚═╝  ╚═╝  
"""

import numpy as np

# from ..._tensor import Tensor, einsum, matmul, f, δ, Δ, Δδ
from .._math_tensor import trace, ddot, sqrt
from .. import _math_array as array
from .._linalg import _linalg_tensor as linalg


def dev(A):
    "Deviatoric Part of a Tensor."
    dim = A.shape[0]
    return A - trace(A) / dim * array.eye(A)


def tresca(A):
    "Tresca Invariant."
    λ = linalg.eigvalsh(A)
    return (λ[-1] - λ[0]) / 2


def von_mises(A):
    "Von Mises Invariant."
    a = dev(A)
    return sqrt(3 / 2 * ddot(a, a))


def triu_1d(A):
    "Flattened upper triangle entries of a Tensor."
    return A[np.triu_indices(A.shape[0])]


def from_triu_1d(A):
    "Recover Tensor from upper triangle entries of a Tensor."
    size_from_dim = np.array([d**2 / 2 + d / 2 for d in np.arange(4)], dtype=int)
    dim = np.where(size_from_dim == A.size)[0][0]
    idx = np.zeros((dim, dim), dtype=int)
    idx.T[np.triu_indices(dim)] = idx[np.triu_indices(dim)] = np.arange(A.size)
    return A[idx.ravel()].reshape(dim, dim)
