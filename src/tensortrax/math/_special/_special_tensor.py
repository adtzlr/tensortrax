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

from .. import _math_array as array
from .._linalg import _linalg_tensor as linalg
from .._math_tensor import einsum, sqrt, trace, transpose


def ddot(A, B):
    return einsum("ij...,ij...->...", A, B)


def dev(A):
    "Deviatoric part of a Tensor."
    dim = A.shape[0]
    return A - trace(A) / dim * array.eye(A)


def sym(A):
    "Symmetric part of a Tensor."
    return (A + transpose(A)) / 2


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


def _from_triu_helper(A):
    size_from_dim = np.array(
        [np.sum(1 + np.arange(d)) for d in np.arange(4)], dtype=int
    )
    size = A.shape[0]
    dim = np.where(size_from_dim == size)[0][0]
    idx = np.zeros((dim, dim), dtype=int)
    idx.T[np.triu_indices(dim)] = idx[np.triu_indices(dim)] = np.arange(size)
    return idx, dim


def from_triu_1d(A):
    "Recover full Tensor from upper triangle entries of a Tensor."
    idx, dim = _from_triu_helper(A)
    return sym(A[idx.ravel()].reshape(dim, dim, *A.shape[1:]))


def from_triu_2d(A):
    "Recover full Tensor from upper triangle entries of a Tensor."
    idx, dim = _from_triu_helper(A)
    return A[idx.ravel()][:, idx.ravel()].reshape(dim, dim, dim, dim, *A.shape[2:])
