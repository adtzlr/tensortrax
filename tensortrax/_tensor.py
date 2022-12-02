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

from ._helpers import f, δ, Δ, Δδ


class Tensor:
    """A (hyper-dual) Tensor with trailing axes.

    Attributes
    ----------
    x : array_like
        The data of the tensor.
    δx : arrayl_like or None
        (Dual) variation data (δ-operator) of the tensor.
    Δx : arrayl_like or None
        (Dual) variation data (Δ-operator) of the tensor.
    Δδx : arrayl_like or None
        (Dual) linearization data (Δδ-operator) of the tensor.
    ntrax : int
        Number of trailing axes.
    shape : tuple
        Shape of the tensor (without trailing axes).
    trax : tuple
        Shape of the trailing axes.
    size : int
        Product of shape (without trailing axes) of the tensor.

    """

    def __init__(self, x, δx=None, Δx=None, Δδx=None, ntrax=2):
        """Init a Hyper-Dual Tensor with trailing axes.

        Parameters
        ----------
        x : array_like
            The data of the tensor.
        δx : arrayl_like or None, optional
            (Dual) variation data (δ-operator) of the tensor.
        Δx : arrayl_like or None, optional
            (Dual) variation data (Δ-operator) of the tensor.
        Δδx : arrayl_like or None, optional
            (Dual) linearization data (Δδ-operator) of the tensor.
        ntrax : int, optional
            Number of trailing axes (default is 2).

        """

        self.x = np.asarray(x)

        self.ntrax = ntrax
        self.shape = x.shape[: len(x.shape) - ntrax]
        self.trax = x.shape[len(x.shape) - ntrax :]
        self.size = np.product(self.shape, dtype=int)

        self.δx = self._init_and_reshape(δx)
        self.Δx = self._init_and_reshape(Δx)
        self.Δδx = self._init_and_reshape(Δδx)

    def _init_and_reshape(self, value):
        if value is None:
            value = np.zeros(self.shape)
        else:
            value = np.asarray(value)
        if len(value.shape) != len(self.x.shape):
            value = value.reshape([*self.shape, *np.ones(self.ntrax, dtype=int)])
        return value

    def __neg__(self):
        return Tensor(x=-self.x, δx=-self.δx, Δx=-self.Δx, Δδx=-self.Δδx)

    def __add__(self, B):
        A = self
        if isinstance(B, Tensor):
            x = f(A) + f(B)
            δx = δ(A) + δ(B)
            Δx = Δ(A) + Δ(B)
            Δδx = Δδ(A) + Δδ(B)
        else:
            x = f(A) + B
            δx = δ(A)
            Δx = Δ(A)
            Δδx = Δδ(A)
        return Tensor(x=x, δx=δx, Δx=Δx, Δδx=Δδx)

    def __sub__(self, B):
        A = self
        if isinstance(B, Tensor):
            x = f(A) - f(B)
            δx = δ(A) - δ(B)
            Δx = Δ(A) - Δ(B)
            Δδx = Δδ(A) - Δδ(B)
        else:
            x = f(A) - B
            δx = δ(A)
            Δx = Δ(A)
            Δδx = Δδ(A)
        return Tensor(x=x, δx=δx, Δx=Δx, Δδx=Δδx)

    def __mul__(self, B):
        A = self
        if isinstance(B, Tensor):
            x = f(A) * f(B)
            δx = δ(A) * f(B) + f(A) * δ(B)
            Δx = Δ(A) * f(B) + f(A) * Δ(B)
            Δδx = Δ(A) * δ(B) + δ(A) * Δ(B) + Δδ(A) * f(B) + f(A) * Δδ(B)
        else:
            x = f(A) * B
            δx = δ(A) * B
            Δx = Δ(A) * B
            Δδx = Δδ(A) * B
        return Tensor(x=x, δx=δx, Δx=Δx, Δδx=Δδx)

    def __truediv__(self, B):
        A = self
        if isinstance(B, Tensor):
            raise NotImplementedError("Divide by Tensor is not supported.")
        else:
            return Tensor(x=f(A) / B, δx=δ(A) / B, Δx=Δ(A) / B, Δδx=Δδ(A) / B)

    def __pow__(self, p):
        A = self
        x = f(A) ** p
        δx = p * f(A) ** (p - 1) * δ(A)
        Δx = p * f(A) ** (p - 1) * Δ(A)
        Δδx = p * f(A) ** (p - 1) * Δδ(A) + p * (p - 1) * f(A) ** (p - 2) * δ(A) * Δ(A)
        return Tensor(x=x, δx=δx, Δx=Δx, Δδx=Δδx)

    def T(self):
        return transpose(self)

    def __matmul__(self, B):
        return matmul(self, B)

    __radd__ = __add__
    __rsub__ = __sub__
    __rmul__ = __mul__
    __rmatmul__ = __matmul__
    __rtruediv__ = __truediv__


def einsum2(subscripts, *operands):
    "Einsum with two operands."
    A, B = operands
    _einsum = lambda *operands: np.einsum(subscripts, *operands)
    if isinstance(A, Tensor):
        if isinstance(B, Tensor):
            x = _einsum(f(A), f(B))
            δx = _einsum(δ(A), f(B)) + _einsum(f(A), δ(B))
            Δx = _einsum(Δ(A), f(B)) + _einsum(f(A), Δ(B))
            Δδx = (
                _einsum(Δδ(A), f(B))
                + _einsum(f(A), Δδ(B))
                + _einsum(δ(A), Δ(B))
                + _einsum(Δ(A), δ(B))
            )
        else:
            x = _einsum(f(A), B)
            δx = _einsum(δ(A), B)
            Δx = _einsum(Δ(A), B)
            Δδx = _einsum(Δδ(A), B)
        return Tensor(x=x, δx=δx, Δx=Δx, Δδx=Δδx)
    else:
        return _einsum(*operands)


def einsum1(subscripts, *operands):
    "Einsum with one operand."
    A = operands[0]
    _einsum = lambda *operands: np.einsum(subscripts, *operands)
    if isinstance(A, Tensor):
        x = _einsum(f(A))
        δx = _einsum(δ(A))
        Δx = _einsum(Δ(A))
        Δδx = _einsum(Δδ(A))
        return Tensor(x=x, δx=δx, Δx=Δx, Δδx=Δδx)
    else:
        return _einsum(*operands)


def einsum(subscripts, *operands):
    "Einsum limited to one or two operands."
    if len(operands) == 1:
        return einsum1(subscripts, *operands)
    elif len(operands) == 2:
        return einsum2(subscripts, *operands)
    else:
        raise NotImplementedError("More than two operands are not supported.")


def transpose(A):
    return einsum("ij...->ji...", A)


def matmul(A, B):
    return einsum("ik...,kj...->ij...", A, B)
