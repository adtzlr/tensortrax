"""
tensorTRAX: Math on (Hyper-Dual) Tensors with Trailing Axes.
"""


import numpy as np

from ..._tensor import Tensor, Δ, Δδ, einsum, f, matmul, δ
from .._math_tensor import exp, sum, transpose
from ..special._special_tensor import ddot
from . import _linalg_array as linalg

dot = matmul


def det(A):
    "Determinant of a 2x2 or 3x3 Tensor."
    if isinstance(A, Tensor):
        x = linalg.det(f(A))
        B = transpose(linalg.inv(f(A)))
        δx = x * ddot(B, δ(A))
        Δx = x * ddot(B, Δ(A))

        ΔB = -matmul(matmul(B, transpose(Δ(A))), B)
        Δδx = Δx * δx / x + x * ddot(ΔB, δ(A)) + x * ddot(B, Δδ(A))
        return Tensor(
            x=x,
            δx=δx,
            Δx=Δx,
            Δδx=Δδx,
            ntrax=A.ntrax,
        )
    else:
        return linalg.det(A)


def inv(A, inverse=linalg.inv):
    "Inverse of a 2x2 or 3x3 Tensor."
    if isinstance(A, Tensor):
        x = inverse(f(A))
        invA = inverse(f(A))
        δx = -matmul(matmul(invA, δ(A)), invA)
        Δx = -matmul(matmul(invA, Δ(A)), invA)
        Δδx = -(
            matmul(matmul(Δx, δ(A)), invA)
            + matmul(matmul(invA, δ(A)), Δx)
            + matmul(matmul(invA, Δδ(A)), invA)
        )

        return Tensor(
            x=x,
            δx=δx,
            Δx=Δx,
            Δδx=Δδx,
            ntrax=A.ntrax,
        )
    else:
        return linalg.inv(A)


def pinv(A):
    "Pseudo-Inverse of a 2x2 or 3x3 Tensor."

    return inv(A, inverse=linalg.pinv)


def eigvalsh(A, eps=np.sqrt(np.finfo(float).eps)):
    "Eigenvalues of a symmetric Tensor."

    A[0, 0] += eps
    A[1, 1] -= eps

    λ, N = [x.T for x in np.linalg.eigh(f(A).T)]
    M = einsum("ai...,aj...->aij...", N, N)

    dim = len(λ)

    δλ = einsum("aij...,ij...->a...", M, δ(A))
    Δλ = einsum("aij...,ij...->a...", M, Δ(A))

    # alpha = [0, 1, 2]
    # beta = [(1, 2), (2, 0), (0, 1)]

    alpha = np.arange(dim)
    beta = [
        np.concatenate([np.arange(a + 1, dim), np.arange(a)]) for a in np.arange(dim)
    ]

    δN = []
    for α in alpha:
        δNα = []
        for β in beta[α]:
            Mαβ = einsum("i...,j...->ij...", N[α], N[β])
            δAαβ = einsum("ij...,ij...->...", Mαβ, δ(A))
            λαβ = λ[α] - λ[β]
            δNα.append(1 / λαβ * N[β] * δAαβ)
        δN.append(sum(δNα, axis=0))

    δM = einsum("ai...,aj...->aij...", δN, N) + einsum("ai...,aj...->aij...", N, δN)
    Δδλ = einsum("aij...,ij...->a...", δM, Δ(A)) + einsum(
        "aij...,ij...->a...", M, Δδ(A)
    )

    return Tensor(
        x=λ,
        δx=δλ,
        Δx=Δλ,
        Δδx=Δδλ,
        ntrax=A.ntrax,
    )


def eigh(A, eps=np.sqrt(np.finfo(float).eps)):
    "Eigenvalues and -bases of a symmetric Tensor (only first derivative)."

    A[0, 0] += eps
    A[1, 1] -= eps

    λ, N = [x.T for x in np.linalg.eigh(f(A).T)]
    M = einsum("ai...,aj...->aij...", N, N)

    dim = len(λ)

    δλ = einsum("aij...,ij...->a...", M, δ(A))
    Δλ = einsum("aij...,ij...->a...", M, Δ(A))

    # alpha = [0, 1, 2]
    # beta = [(1, 2), (2, 0), (0, 1)]

    alpha = np.arange(dim)
    beta = [
        np.concatenate([np.arange(a + 1, dim), np.arange(a)]) for a in np.arange(dim)
    ]

    δN = []
    for α in alpha:
        δNα = []
        for β in beta[α]:
            Mαβ = einsum("i...,j...->ij...", N[α], N[β])
            δAαβ = einsum("ij...,ij...->...", Mαβ, δ(A))
            λαβ = λ[α] - λ[β]
            δNα.append(1 / λαβ * N[β] * δAαβ)
        δN.append(sum(δNα, axis=0))

    δM = einsum("ai...,aj...->aij...", δN, N) + einsum("ai...,aj...->aij...", N, δN)
    Δδλ = einsum("aij...,ij...->a...", δM, Δ(A)) + einsum(
        "aij...,ij...->a...", M, Δδ(A)
    )

    return (
        Tensor(
            x=λ,
            δx=δλ,
            Δx=Δλ,
            Δδx=Δδλ,
            ntrax=A.ntrax,
        ),
        Tensor(
            x=M,
            δx=δM,
            Δx=δM * np.nan,
            Δδx=δM * np.nan,
            ntrax=A.ntrax,
        ),
    )


def expm(A):
    "Compute the matrix exponential of a symmetric array."
    λ, M = eigh(A)
    return einsum("a...,aij...->ij...", exp(λ), M)
