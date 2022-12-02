r"""
 _                            
| |                          ████████╗██████╗  █████╗ ██╗  ██╗
| |_ ___ _ __  ___  ___  _ __╚══██╔══╝██╔══██╗██╔══██╗╚██╗██╔╝
| __/ _ \ '_ \/ __|/ _ \| '__|  ██║   ██████╔╝███████║ ╚███╔╝ 
| ||  __/ | | \__ \ (_) | |     ██║   ██╔══██╗██╔══██║ ██╔██╗ 
 \__\___|_| |_|___/\___/|_|     ██║   ██║  ██║██║  ██║██╔╝ ██╗
                                ╚═╝   ╚═╝  ╚═╝╚═╝  ╚═╝╚═╝  ╚═╝  
"""

from joblib import delayed, Parallel, cpu_count
import numpy as np

from .._tensor import Tensor, einsum, matmul, f, δ, Δ, Δδ
from . import _array as array


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
    )


def sin(A):
    if isinstance(A, Tensor):
        return Tensor(
            x=np.sin(f(A)),
            δx=np.cos(f(A)) * δ(A),
            Δx=np.cos(f(A)) * Δ(A),
            Δδx=-np.sin(f(A)) * δ(A) * Δ(A) + np.cos(f(A)) * Δδ(A),
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
        )
    else:
        return np.tanh(A)


def function(fun, ntrax=0, njobs=cpu_count()):
    "Evaluate a scalar-valued function."

    def evaluate_function(x, *args, **kwargs):
        return fun(Tensor(x, ntrax=ntrax), *args, **kwargs).x

    return evaluate_function


def gradient(fun, ntrax=0, njobs=cpu_count()):
    "Evaluate the gradient of a scalar-valued function."

    def evaluate_gradient(x, *args, **kwargs):

        t = Tensor(x, ntrax=ntrax)
        indices = range(t.size)

        fx = np.zeros((1, *t.trax))
        dfdx = np.zeros((t.size, *t.trax))
        δx = np.eye(t.size)

        def kernel(a, x, δx, *args, **kwargs):
            t = Tensor(x, δx=δx[a], Δx=δx[a], ntrax=ntrax)
            func = fun(t, *args, **kwargs)
            fx[:] = f(func)
            dfdx[a] = δ(func)

        Parallel(n_jobs=njobs, backend="threading")(
            delayed(kernel)(a, x, δx, *args, **kwargs) for a in indices
        )

        return np.array(dfdx).reshape(*t.shape, *t.trax), fx[0]

    return evaluate_gradient


def hessian(fun, ntrax=0, njobs=cpu_count()):
    "Evaluate the hessian of a scalar-valued function."

    def evaluate_hessian(x, *args, **kwargs):

        t = Tensor(x, ntrax=ntrax)
        indices = np.array(np.triu_indices(t.size)).T

        fx = np.zeros((1, *t.trax))
        dfdx = np.zeros((t.size, *t.trax))
        d2fdx2 = np.zeros((t.size, t.size, *t.trax))
        δx = Δx = np.eye(t.size)

        def kernel(a, b, x, δx, *args, **kwargs):
            t = Tensor(x, δx=δx[a], Δx=Δx[b], ntrax=ntrax)
            func = fun(t, *args, **kwargs)
            fx[:] = f(func)
            dfdx[a] = δ(func)
            d2fdx2[a, b] = d2fdx2[b, a] = Δδ(func)

        Parallel(n_jobs=njobs, backend="threading")(
            delayed(kernel)(a, b, x, δx, *args, **kwargs) for a, b in indices
        )

        return (
            np.array(d2fdx2).reshape(*t.shape, *t.shape, *t.trax),
            np.array(dfdx).reshape(*t.shape, *t.trax),
            fx[0],
        )

    return evaluate_hessian


def gradient_vector_product(fun, ntrax=0, njobs=cpu_count()):
    "Evaluate the gradient-vector-product of a function."

    def evaluate_gradient_vector_product(x, δx, *args, **kwargs):
        return fun(Tensor(x, δx, ntrax=ntrax), *args, **kwargs).δx

    return evaluate_gradient_vector_product


def hessian_vector_product(fun, ntrax=0, njobs=cpu_count()):
    "Evaluate the gradient-vector-product of a function."

    def evaluate_hessian_vector_product(x, δx, Δx, *args, **kwargs):
        return fun(Tensor(x, δx, Δx, ntrax=ntrax), *args, **kwargs).Δδx

    return evaluate_hessian_vector_product
