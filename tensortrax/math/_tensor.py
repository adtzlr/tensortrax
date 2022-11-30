"""
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
from ._array import det as _det, inv as _inv, eye as _eye


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
        x = _det(f(A))
        B = transpose(_inv(f(A)))
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
        return _det(A)


def eigvalsh(A):
    "Eigenvalues of a symmetric Tensor."

    λ, N = [x.T for x in np.linalg.eigh(f(A).T)]
    M = einsum("ia...,ja...->aij...", N, N)

    δλ = einsum("aij...,ij...->a...", M, δ(A))
    Δλ = einsum("aij...,ij...->a...", M, Δ(A))

    I = _eye(f(A))
    Γ = [(1, 2), (2, 0), (0, 1)]

    δM = []
    for α in range(3):

        δMα = []
        for γ in Γ[α]:

            u = f(A) - λ[α] * I
            v = λ[γ] - λ[α]

            δu = δ(A) - δλ[α] * I
            δv = δλ[γ] - δλ[α]

            δMα.append((δu * v - u * δv) / v**2)

        δM.append(matmul(*δMα))

    Δδλ = einsum("aij...,ij...->a...", δM, Δ(A)) + einsum(
        "aij...,ij...->a...", M, Δδ(A)
    )

    return Tensor(
        x=λ,
        δx=δλ,
        Δx=Δλ,
        Δδx=Δδλ,
    )


def function(fun, ntrax=2):
    "Evaluate a scalar-valued function."

    def evaluate_function(x, *args, **kwargs):
        return fun(Tensor(x, ntrax=ntrax), *args, **kwargs).x

    return evaluate_function


def gradient(fun, ntrax=2, n_jobs=cpu_count()):
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

        Parallel(n_jobs=n_jobs, backend="threading")(
            delayed(kernel)(a, x, δx, *args, **kwargs) for a in indices
        )

        return np.array(dfdx).reshape(*t.shape, *t.trax), fx[0]

    return evaluate_gradient


def hessian(fun, ntrax=2, n_jobs=cpu_count()):
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

        Parallel(n_jobs=n_jobs, backend="threading")(
            delayed(kernel)(a, b, x, δx, *args, **kwargs) for a, b in indices
        )

        return (
            np.array(d2fdx2).reshape(*t.shape, *t.shape, *t.trax),
            np.array(dfdx).reshape(*t.shape, *t.trax),
            fx[0],
        )

    return evaluate_hessian
