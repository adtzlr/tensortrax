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

from .._tensor import Tensor, einsum, matmul, f, δ, Δ, Δδ
from ._linalg import _linalg_array as array


dot = matmul


def ddot(A, B):
    return einsum("ij...,ij...->...", A, B)


def trace(A):
    return einsum("ii...->...", A)


def transpose(A):
    return einsum("ij...->ji...", A)


def sum(A, axis=0):
    if isinstance(A, Tensor):
        return Tensor(
            x=np.sum(f(A), axis=axis),
            δx=np.sum(δ(A), axis=axis),
            Δx=np.sum(Δ(A), axis=axis),
            Δδx=np.sum(Δδ(A), axis=axis),
            ntrax=A.ntrax,
        )
    else:
        return np.sum(A, axis=axis)


def sqrt(A):
    if isinstance(A, Tensor):
        return A**0.5
    else:
        return np.sqrt(A)


def det(A):
    "Determinant of a 2x2 or 3x3 Tensor."
    if isinstance(A, Tensor):
        x = array.det(f(A))
        B = transpose(array.inv(f(A)))
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
        return array.det(A)


def eigvalsh(A):
    "Eigenvalues of a symmetric Tensor."

    λ, N = [x.T for x in np.linalg.eigh(f(A).T)]
    N = transpose(N)
    M = einsum("ai...,aj...->aij...", N, N)

    δλ = einsum("aij...,ij...->a...", M, δ(A))
    Δλ = einsum("aij...,ij...->a...", M, Δ(A))

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
    Δδλ = einsum("aij...,ij...->a...", δM, Δ(A)) + einsum(
        "aij...,ij...->a...", M, Δδ(A)
    )

    # λ_equal = np.isclose(sum(λ, axis=0), 3)
    # Δδλ[..., λ_equal] = np.trace(Δδ(A))[λ_equal] / 3

    return Tensor(
        x=λ,
        δx=δλ,
        Δx=Δλ,
        Δδx=Δδλ,
        ntrax=A.ntrax,
    )


def sin(A):
    if isinstance(A, Tensor):
        return Tensor(
            x=np.sin(f(A)),
            δx=np.cos(f(A)) * δ(A),
            Δx=np.cos(f(A)) * Δ(A),
            Δδx=-np.sin(f(A)) * δ(A) * Δ(A) + np.cos(f(A)) * Δδ(A),
            ntrax=A.ntrax,
        )
    else:
        return np.sin(A)


def cos(A):
    if isinstance(A, Tensor):
        return Tensor(
            x=np.cos(f(A)),
            δx=-np.sin(f(A)) * δ(A),
            Δx=-np.sin(f(A)) * Δ(A),
            Δδx=-np.cos(f(A)) * δ(A) * Δ(A) - np.sin(f(A)) * Δδ(A),
            ntrax=A.ntrax,
        )
    else:
        return np.cos(A)


def tan(A):
    if isinstance(A, Tensor):
        return Tensor(
            x=np.tan(f(A)),
            δx=np.cos(f(A)) ** -2 * δ(A),
            Δx=np.cos(f(A)) ** -2 * Δ(A),
            Δδx=2 * np.tan(f(A)) * np.cos(f(A)) ** -2 * δ(A) * Δ(A)
            + np.cos(f(A)) ** -2 * Δδ(A),
            ntrax=A.ntrax,
        )
    else:
        return np.tan(A)


def tanh(A):
    if isinstance(A, Tensor):
        x = np.tanh(f(A))
        return Tensor(
            x=x,
            δx=(1 - x**2) * δ(A),
            Δx=(1 - x**2) * Δ(A),
            Δδx=-2 * x * (1 - x**2) * δ(A) * Δ(A) + (1 - x**2) * Δδ(A),
            ntrax=A.ntrax,
        )
    else:
        return np.tanh(A)
    