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

from ._helpers import f, δ


class Tensor:
    """A (hyper-dual) Tensor with trailing axes.

    Attributes
    ----------
    x : array_like
        The data of the tensor.
    δx : arrayl_like or None
        (Dual) variation data (δ-operator) of the tensor.
    ntrax : int
        Number of trailing axes.
    shape : tuple
        Shape of the tensor (without trailing axes).
    trax : tuple
        Shape of the trailing axes.
    size : int
        Product of shape (without trailing axes) of the tensor.

    """

    def __init__(self, x, δx=None, Δx=None, Δδx=None, ntrax=0):
        """Init a Hyper-Dual Tensor with trailing axes.

        Parameters
        ----------
        x : array_like
            The data of the tensor.
        δx : arrayl_like or None, optional
            (Dual) variation data (δ-operator) of the tensor.
        ntrax : int, optional
            Number of trailing axes (default is 2).

        """

        self.x = np.asarray(x)

        self.ntrax = ntrax
        self.shape = x.shape[: len(x.shape) - ntrax]
        self.trax = x.shape[len(x.shape) - ntrax :]
        self.size = np.product(self.shape, dtype=int)

        self.δx = self._init_and_reshape(δx)

    def _init_and_reshape(self, value):
        if value is None:
            value = np.zeros(self.shape)
        else:
            value = np.asarray(value)
        if len(value.shape) != len(self.x.shape):
            value = value.reshape([*self.shape, *np.ones(self.ntrax, dtype=int)])
        return value

    def __neg__(self):
        return Tensor(x=-self.x, δx=-self.δx, ntrax=self.ntrax)

    def __add__(self, other):
        A = self
        B = astensor(other)
        return Tensor(x=f(A) + f(B), δx=δ(A) + δ(B), ntrax=A.ntrax)

    def __sub__(self, other):
        A = self
        B = astensor(other)
        return Tensor(x=f(A) - f(B), δx=δ(A) - δ(B), ntrax=A.ntrax)

    def __rsub__(self, other):
        return -self.__sub__(other)

    def __mul__(self, other):
        A = self
        B = astensor(other)
        mul = lambda A, B: self.__mul__(B) if istensor(B) else A * B
        x = mul(f(A), f(B))
        δx = mul(δ(A), f(B)) + mul(f(A), δ(B))
        return Tensor(x=x, δx=δx, ntrax=A.ntrax)

    def __truediv__(self, B):
        A = self
        return A * B**-1

    def __rtruediv__(self, B):
        A = self
        return B * A**-1

    def __pow__(self, p):
        A = self
        x = f(A) ** p
        δx = p * f(A) ** (p - 1) * δ(A)
        return Tensor(x=x, δx=δx, ntrax=A.ntrax)

    def __gt__(self, B):
        A = self
        return f(A) > (f(B) if istensor(B) else B)

    def __lt__(self, B):
        A = self
        return f(A) < (f(B) if istensor(B) else B)

    def __ge__(self, B):
        A = self
        return f(A) >= (f(B) if istensor(B) else B)

    def __le__(self, B):
        A = self
        return f(A) <= (f(B) if istensor(B) else B)

    def __eq__(self, B):
        A = self
        return f(A) == (f(B) if istensor(B) else B)

    def __matmul__(self, other):
        A = self
        B = astensor(other)
        return matmul(A, B)

    def __rmatmul__(self, other):
        A = self
        B = astensor(other)
        return matmul(B, A)

    def __getitem__(self, key):
        x = f(self)[key]
        δx = δ(self)[key]
        return Tensor(x=x, δx=δx, ntrax=self.ntrax)

    def __setitem__(self, key, value):
        if isinstance(value, Tensor):
            self.x[key] = f(value)
            if self.δx[key].shape != δ(value).shape:
                self.δx = np.tile(
                    self.δx.reshape(*self.shape, *np.ones(len(self.trax), dtype=int)),
                    (*np.ones(len(self.shape), dtype=int), *self.trax),
                )
            self.δx[key] = δ(value)
        else:
            self.x[key] = value
            self.δx[key].fill(0)

    def __repr__(self):
        header = "<tensortrax tensor object>"
        metadata = [
            f"  Shape / size of trailing axes: {self.trax} / {self.ntrax}",
            f"  Shape / size of tensor: {self.shape} / {self.size}",
        ]
        data = self.x.__repr__()
        return "\n".join([header, *metadata, "", data])

    def T(self):
        return transpose(self)

    def ravel(self, order="C"):
        return ravel(self, order=order)

    def reshape(self, *shape, order="C"):
        return reshape(self, newshape=shape, order=order)

    __radd__ = __add__
    __rmul__ = __mul__
    __array_ufunc__ = None


def istensor(A):
    return isinstance(A, Tensor)


def astensor(A):
    return A if istensor(A) else Tensor(A)


def ravel(A, order="C"):
    if istensor(A):
        δtrax = δ(A).shape[len(A.shape) :]
        return Tensor(
            x=f(A).reshape(A.size, *A.trax, order=order),
            δx=δ(A).reshape(A.size, *δtrax, order=order),
            ntrax=A.ntrax,
        )
    else:
        return np.ravel(A, order=order)


def reshape(A, newshape, order="C"):
    if istensor(A):
        δtrax = δ(A).shape[len(A.shape) :]
        return Tensor(
            x=f(A).reshape(*newshape, *A.trax, order=order),
            δx=δ(A).reshape(*newshape, *δtrax, order=order),
            ntrax=A.ntrax,
        )
    else:
        return np.reshape(A, newshape=newshape, order=order)


def einsum3(subscripts, *operands):
    "Einsum with three operands."
    A, B, C = operands
    _einsum = lambda *operands: np.einsum(subscripts, *operands)

    if istensor(A) and istensor(B) and istensor(C):
        x = _einsum(f(A), f(B), f(C))
        δx = (
            _einsum(δ(A), f(B), f(C))
            + _einsum(f(A), δ(B), f(C))
            + _einsum(f(A), f(B), δ(C))
        )
        ntrax = A.ntrax
    elif istensor(A) and not istensor(B) and not istensor(C):
        x = _einsum(f(A), B, C)
        δx = _einsum(δ(A), B, C)
        ntrax = A.ntrax
    elif not istensor(A) and istensor(B) and not istensor(C):
        x = _einsum(A, f(B), C)
        δx = _einsum(A, δ(B), C)
        ntrax = B.ntrax
    elif not istensor(A) and not istensor(B) and istensor(C):
        x = _einsum(A, B, f(C))
        δx = _einsum(A, B, δ(C))
        ntrax = C.ntrax
    elif istensor(A) and istensor(B) and not istensor(C):
        x = _einsum(f(A), f(B), C)
        δx = _einsum(δ(A), f(B), C) + _einsum(f(A), δ(B), C)
        ntrax = A.ntrax
    elif istensor(A) and not istensor(B) and istensor(C):
        x = _einsum(f(A), B, f(C))
        δx = _einsum(δ(A), B, f(C)) + _einsum(f(A), B, δ(C))
        ntrax = A.ntrax
    elif not istensor(A) and istensor(B) and istensor(C):
        x = _einsum(A, f(B), f(C))
        δx = _einsum(A, δ(B), f(C)) + _einsum(A, f(B), δ(C))
        ntrax = B.ntrax
    else:
        return _einsum(*operands)

    return Tensor(x=x, δx=δx, ntrax=ntrax)


def einsum2(subscripts, *operands):
    "Einsum with two operands."
    A, B = operands
    _einsum = lambda *operands: np.einsum(subscripts, *operands)

    if istensor(A) and istensor(B):
        x = _einsum(f(A), f(B))
        δx = _einsum(δ(A), f(B)) + _einsum(f(A), δ(B))
        ntrax = A.ntrax
    elif istensor(A) and not istensor(B):
        x = _einsum(f(A), B)
        δx = _einsum(δ(A), B)
        ntrax = A.ntrax
    elif not istensor(A) and istensor(B):
        x = _einsum(A, f(B))
        δx = _einsum(A, δ(B))
        ntrax = B.ntrax
    else:
        return _einsum(*operands)

    return Tensor(x=x, δx=δx, ntrax=ntrax)


def einsum1(subscripts, *operands):
    "Einsum with one operand."
    A = astensor(operands[0])
    _einsum = (
        lambda subscripts, *operands: einsum1(subscripts, *operands)
        if istensor(operands[0])
        else np.einsum(subscripts, *operands)
    )
    x = _einsum(subscripts, f(A))
    δx = _einsum(subscripts, δ(A))
    return Tensor(x=x, δx=δx, ntrax=A.ntrax)


def einsum(subscripts, *operands):
    "Einsum limited to one, two or three operands."
    if len(operands) == 1:
        return einsum1(subscripts, *operands)
    elif len(operands) == 2:
        return einsum2(subscripts, *operands)
    elif len(operands) == 3:
        return einsum3(subscripts, *operands)
    else:
        raise NotImplementedError("More than three operands are not supported.")


def transpose(A, axes=None):
    ij = "abcdefghijklmnopqrstuvwxyz"[: len(A.shape)]
    if axes is None:
        ji = ij[::-1]
    else:
        ji = np.array(ij)[axes].tolist()
    return einsum(f"{ij}...->{ji}...", A)


def matmul(A, B):

    A = astensor(A)
    B = astensor(B)

    ik = "ik"[2 - len(A.shape) :]
    kj = "kj"[: len(B.shape)]
    ij = (ik + kj).replace("k", "")

    mmul = (
        lambda A, B: matmul(A, B)
        if (istensor(A) or istensor(B))
        else np.einsum(f"{ik}...,{kj}...->{ij}...", A, B)
    )

    return Tensor(
        x=mmul(f(A), f(B)), δx=mmul(f(A), δ(B)) + mmul(δ(A), f(B)), ntrax=A.ntrax
    )
