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

from ..._tensor import Tensor, einsum, f, matmul, δ
from .._math_tensor import exp, sum, transpose
from .._special._special_tensor import ddot
from . import _linalg_array as array

dot = matmul


def det(A):
    "Determinant of a 2x2 or 3x3 Tensor."
    if isinstance(A, Tensor):
        x = array.det(f(A))
        B = transpose(array.inv(f(A)))
        δx = x * ddot(B, δ(A))
        return Tensor(
            x=x,
            δx=δx,
            ntrax=A.ntrax,
        )
    else:
        return array.det(A)


def inv(A):
    "Inverse of a 2x2 or 3x3 Tensor."
    if isinstance(A, Tensor):
        x = array.inv(f(A))
        invA = array.inv(f(A))
        δx = -matmul(matmul(invA, δ(A)), invA)

        return Tensor(
            x=x,
            δx=δx,
            ntrax=A.ntrax,
        )
    else:
        return array.inv(A)


def eigvalsh(A):
    "Eigenvalues of a symmetric Tensor."

    λ, N = [x.T for x in np.linalg.eigh(f(A).T)]
    M = einsum("ai...,aj...->aij...", N, N)
    δλ = einsum("aij...,ij...->a...", M, δ(A))

    return Tensor(
        x=λ,
        δx=δλ,
        ntrax=A.ntrax,
    )


def eigh(A):
    "Eigenvalues and -bases of a symmetric Tensor (only first derivative)."

    λ, N = [x.T for x in np.linalg.eigh(f(A).T)]
    M = einsum("ai...,aj...->aij...", N, N)
    δλ = einsum("aij...,ij...->a...", M, δ(A))

    Γ = [(1, 2), (2, 0), (0, 1)]

    δN = []
    for α in range(3):
        δNα = []
        for γ in Γ[α]:
            Mαγ = einsum("i...,j...->ij...", N[α], N[γ])
            δAαγ = einsum("ij...,ij...->...", Mαγ, δ(A))
            λαγ = λ[α] - λ[γ]
            λ_equal = np.isclose(λ[α], λ[γ])
            if np.any(λ_equal):
                if len(λαγ.shape) == 0:
                    λαγ = np.inf
                else:
                    λαγ[λ_equal] = np.inf
            δNα.append(1 / λαγ * N[γ] * δAαγ)
        δN.append(sum(δNα, axis=0))

    δM = einsum("ai...,aj...->aij...", δN, N) + einsum("ai...,aj...->aij...", N, δN)

    return (
        Tensor(
            x=λ,
            δx=δλ,
            ntrax=A.ntrax,
        ),
        Tensor(
            x=M,
            δx=δM,
            ntrax=A.ntrax,
        ),
    )


def expm(A):
    "Compute the matrix exponential of a symmetric array."
    λ, M = eigh(A)
    return einsum("a...,aij...->ij...", exp(λ), M)
