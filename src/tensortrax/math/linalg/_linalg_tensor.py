"""
tensorTRAX: Math on (Hyper-Dual) Tensors with Trailing Axes.
"""

import numpy as np

from ..._tensor import Tensor, Δ, Δδ, einsum, f, matmul, δ
from .._math_tensor import exp, sqrt, sum, transpose
from ..special._special_tensor import ddot
from . import _linalg_array as linalg

dot = matmul


# Perturbation to separate repeated eigenvalues. A fixed perturbation can't separate
# a repeated eigenvalue if its eigenspace is a plane on which the perturbation is
# isotropic. For diag(1, -1, 0), these are the planes with normals (1, +/-1, 0), e.g.
# uniaxial loading at 45° in the xy-plane. This generic matrix with eigenvalues
# (-1, 0, 1) moves these planes to directions without any symmetry.
PERTURBATION = np.array(
    [
        [0.03, 0.44, -0.66],
        [0.44, 0.55, -0.24],
        [-0.66, -0.24, -0.58],
    ]
)


def perturb(x, eps):
    "Add a relative perturbation (scaled by the norm of x) to x, in-place."
    dim = x.shape[0]
    norm = np.sqrt(np.einsum("ij...,ij...->...", x, x))
    scale = eps * np.where(norm > 0, norm, 1.0)
    x += scale * PERTURBATION[:dim, :dim].reshape(dim, dim, *np.ones(x.ndim - 2, int))


def det(A):
    "Determinant of a 2x2 or 3x3 Tensor."
    if isinstance(A, Tensor):
        x = linalg.det(f(A))
        cof = transpose(linalg.adj(f(A)))  # dxdA
        Δcof = transpose(linalg.adj_variation(f(A), Δ(A)))  # Δ(dxdA)
        δx = ddot(cof, δ(A))
        Δx = ddot(cof, Δ(A))
        Δδx = ddot(Δcof, δ(A)) + ddot(cof, Δδ(A))
        return Tensor(x=x, δx=δx, Δx=Δx, Δδx=Δδx, ntrax=A.ntrax)
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

    if isinstance(A, Tensor):

        # perturb a copy of the values to separate repeated eigenvalues
        # (the argument and its dual data must not be modified)
        x = f(A).copy()
        perturb(x, eps)

        λ, N = [y.T for y in np.linalg.eigh(x.T)]
        M = einsum("ai...,aj...->aij...", N, N)

        dim = len(λ)

        δλ = einsum("aij...,ij...->a...", M, δ(A))
        Δλ = einsum("aij...,ij...->a...", M, Δ(A))

        # alpha = [0, 1, 2]
        # beta = [(1, 2), (2, 0), (0, 1)]

        alpha = np.arange(dim)
        beta = [
            np.concatenate([np.arange(a + 1, dim), np.arange(a)])
            for a in np.arange(dim)
        ]

        δN = []
        for α in alpha:
            δNα = []
            for β in beta[α]:
                Mαβ = einsum("i...,j...->ij...", N[α], N[β])
                Mαβ = (Mαβ + transpose(Mαβ)) / 2  # A is symmetric -> only sym. variations
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

    else:
        return np.linalg.eigvalsh(A.T).T


def eigh(A, eps=np.sqrt(np.finfo(float).eps)):
    "Eigenvalues and -bases of a symmetric Tensor."

    if isinstance(A, Tensor):
        # perturb a copy of the values to separate repeated eigenvalues
        # (the argument and its dual data must not be modified)
        x = f(A).copy()
        perturb(x, eps)

        λ, N = [y.T for y in np.linalg.eigh(x.T)]
        M = einsum("ai...,aj...->aij...", N, N)

        dim = len(λ)

        δλ = einsum("aij...,ij...->a...", M, δ(A))
        Δλ = einsum("aij...,ij...->a...", M, Δ(A))

        # alpha = [0, 1, 2]
        # beta = [(1, 2), (2, 0), (0, 1)]
        alpha = np.arange(dim)
        beta = [
            np.concatenate([np.arange(a + 1, dim), np.arange(a)])
            for a in np.arange(dim)
        ]

        δN = []
        ΔN = []
        for α in alpha:
            δNα = []
            ΔNα = []
            for β in beta[α]:
                Mαβ = einsum("i...,j...->ij...", N[α], N[β])
                Mαβ = (Mαβ + transpose(Mαβ)) / 2  # A is symmetric -> only sym. variations
                δAαβ = einsum("ij...,ij...->...", Mαβ, δ(A))
                ΔAαβ = einsum("ij...,ij...->...", Mαβ, Δ(A))
                λαβ = λ[α] - λ[β]
                δNα.append(1 / λαβ * N[β] * δAαβ)
                ΔNα.append(1 / λαβ * N[β] * ΔAαβ)
            δN.append(sum(δNα, axis=0))
            ΔN.append(sum(ΔNα, axis=0))

        ΔδN = []
        δAsym = (δ(A) + transpose(δ(A))) / 2
        for α in alpha:
            ΔδNα = []
            for β in beta[α]:
                Mαβ = einsum("i...,j...->ij...", N[α], N[β])
                Mαβ = (Mαβ + transpose(Mαβ)) / 2  # A is symmetric -> only sym. variations
                ΔMαβ = einsum("i...,j...->ij...", ΔN[α], N[β]) + einsum(
                    "i...,j...->ij...", N[α], ΔN[β]
                )
                δAαβ = einsum("ij...,ij...->...", Mαβ, δ(A))
                ΔδAαβ = einsum("ij...,ij...->...", ΔMαβ, δAsym) + einsum(
                    "ij...,ij...->...", Mαβ, Δδ(A)
                )
                λαβ = λ[α] - λ[β]
                Δλαβ = Δλ[α] - Δλ[β]
                ΔδNα.append(
                    -(λαβ**-2) * Δλαβ * N[β] * δAαβ
                    + 1 / λαβ * ΔN[β] * δAαβ
                    + 1 / λαβ * N[β] * ΔδAαβ
                )
            ΔδN.append(sum(ΔδNα, axis=0))

        δM = einsum("ai...,aj...->aij...", δN, N) + einsum("ai...,aj...->aij...", N, δN)
        ΔM = einsum("ai...,aj...->aij...", ΔN, N) + einsum("ai...,aj...->aij...", N, ΔN)
        Δδλ = einsum("aij...,ij...->a...", δM, Δ(A)) + einsum(
            "aij...,ij...->a...", M, Δδ(A)
        )

        ΔδM = (
            einsum("ai...,aj...->aij...", δN, ΔN)
            + einsum("ai...,aj...->aij...", ΔN, δN)
            + einsum("ai...,aj...->aij...", ΔδN, N)
            + einsum("ai...,aj...->aij...", N, ΔδN)
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
                Δx=ΔM,
                Δδx=ΔδM,
                ntrax=A.ntrax,
            ),
        )

    else:
        λ, N = [x.T for x in np.linalg.eigh(A.T)]
        M = einsum("ai...,aj...->aij...", N, N)
        return λ, M


def expm(A):
    "Compute the matrix exponential of a symmetric array."
    λ, M = eigh(A)
    return einsum("a...,aij...->ij...", exp(λ), M)


def sqrtm(A):
    "Compute the matrix square root of a symmetric array."
    λ, M = eigh(A)
    return einsum("a...,aij...->ij...", sqrt(λ), M)
