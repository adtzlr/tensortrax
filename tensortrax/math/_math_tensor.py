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

from .._tensor import Tensor, Δ, Δδ, einsum, f, matmul, ravel, reshape, δ
from ._linalg import _linalg_array as array

dot = matmul


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


def sinh(A):
    if isinstance(A, Tensor):
        return Tensor(
            x=np.sinh(f(A)),
            δx=np.cosh(f(A)) * δ(A),
            Δx=np.cosh(f(A)) * Δ(A),
            Δδx=np.sinh(f(A)) * δ(A) * Δ(A) + np.cosh(f(A)) * Δδ(A),
            ntrax=A.ntrax,
        )
    else:
        return np.sinh(A)


def cosh(A):
    if isinstance(A, Tensor):
        return Tensor(
            x=np.cosh(f(A)),
            δx=np.sinh(f(A)) * δ(A),
            Δx=np.sinh(f(A)) * Δ(A),
            Δδx=np.cosh(f(A)) * δ(A) * Δ(A) + np.sinh(f(A)) * Δδ(A),
            ntrax=A.ntrax,
        )
    else:
        return np.cosh(A)


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


def exp(A):
    if isinstance(A, Tensor):
        x = np.exp(f(A))
        return Tensor(
            x=x,
            δx=x * δ(A),
            Δx=x * Δ(A),
            Δδx=x * δ(A) * Δ(A) + x * Δδ(A),
            ntrax=A.ntrax,
        )
    else:
        return np.exp(A)


def log(A):
    if isinstance(A, Tensor):
        x = np.log(f(A))
        return Tensor(
            x=x,
            δx=1 / f(A) * δ(A),
            Δx=1 / f(A) * Δ(A),
            Δδx=-1 / f(A) ** 2 * δ(A) * Δ(A) + 1 / f(A) * Δδ(A),
            ntrax=A.ntrax,
        )
    else:
        return np.log(A)


def log10(A):
    if isinstance(A, Tensor):
        x = np.log10(f(A))
        return Tensor(
            x=x,
            δx=1 / (np.log(10) * f(A)) * δ(A),
            Δx=1 / (np.log(10) * f(A)) * Δ(A),
            Δδx=-1 / (np.log(10) * f(A) ** 2) * δ(A) * Δ(A)
            + 1 / (np.log(10) * f(A)) * Δδ(A),
            ntrax=A.ntrax,
        )
    else:
        return np.log10(A)


def diagonal(A, offset=0, axis1=0, axis2=1):
    kwargs = dict(offset=offset, axis1=axis1, axis2=axis2)
    if isinstance(A, Tensor):
        return Tensor(
            x=np.diagonal(f(A), **kwargs).T,
            δx=np.diagonal(δ(A), **kwargs).T,
            Δx=np.diagonal(Δ(A), **kwargs).T,
            Δδx=np.diagonal(Δδ(A), **kwargs).T,
            ntrax=A.ntrax,
        )
    else:
        return np.diagonal(A, **kwargs).T
