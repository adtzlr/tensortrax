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

from .._tensor import Tensor, einsum, f, matmul, ravel, reshape, δ

dot = matmul


def trace(A):
    return einsum("ii...->...", A)


def transpose(A, axes=None):
    ij = "abcdefghijklmnopqrstuvwxyz"[: len(A.shape)]
    if axes is None:
        ji = ij[::-1]
    else:
        ji = np.array(ij)[axes].tolist()
    return einsum(f"{ij}...->{ji}...", A)


def sum(A, axis=0):
    if isinstance(A, Tensor):
        return Tensor(
            x=np.sum(f(A), axis=axis),
            δx=np.sum(δ(A), axis=axis),
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
            ntrax=A.ntrax,
        )
    else:
        return np.sin(A)


def cos(A):
    if isinstance(A, Tensor):
        return Tensor(
            x=np.cos(f(A)),
            δx=-np.sin(f(A)) * δ(A),
            ntrax=A.ntrax,
        )
    else:
        return np.cos(A)


def tan(A):
    if isinstance(A, Tensor):
        return Tensor(
            x=np.tan(f(A)),
            δx=np.cos(f(A)) ** -2 * δ(A),
            ntrax=A.ntrax,
        )
    else:
        return np.tan(A)


def sinh(A):
    if isinstance(A, Tensor):
        return Tensor(
            x=np.sinh(f(A)),
            δx=np.cosh(f(A)) * δ(A),
            ntrax=A.ntrax,
        )
    else:
        return np.sinh(A)


def cosh(A):
    if isinstance(A, Tensor):
        return Tensor(
            x=np.cosh(f(A)),
            δx=np.sinh(f(A)) * δ(A),
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
            ntrax=A.ntrax,
        )
    else:
        return np.diagonal(A, **kwargs).T
