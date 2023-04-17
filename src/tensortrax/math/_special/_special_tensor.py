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

from ..._tensor import Tensor
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
    "Helper to recover full Tensor from upper triangle entries of a Tensor."
    # length of triu-entries
    # size_from_dim = [0, 1, 3, 6]
    # size_from_dim[2] = 3
    # size_from_dim[3] = 6
    # ...
    size_from_dim = np.array(
        [len(np.triu_indices(d)[0]) for d in np.arange(4)], dtype=int
    )
    size = A.shape[0]
    dim = np.where(size_from_dim == size)[0][0]
    idx = np.zeros((dim, dim), dtype=int)
    idx.T[np.triu_indices(dim)] = idx[np.triu_indices(dim)] = np.arange(size)
    return idx, dim


def from_triu_1d(A, like=None):
    "Recover full Tensor from upper triangle entries of a Tensor."
    idx, dim = _from_triu_helper(A)
    out = sym(A[idx.ravel()].reshape(dim, dim, *A.shape[1:]))
    if like is not None:
        axes = len(out.shape)
        if isinstance(A, Tensor):
            axes += out.ntrax
        if axes < (len(like.shape) + like.ntrax):
            ones = np.ones(like.ndual, dtype=int)
            out = out.reshape(dim, dim, *ones, *like.trax)
    return out


def from_triu_2d(A):
    "Recover full Tensor from upper triangle entries of a Tensor."
    idx, dim = _from_triu_helper(A)
    return A[idx.ravel()][:, idx.ravel()].reshape(dim, dim, dim, dim, *A.shape[2:])
