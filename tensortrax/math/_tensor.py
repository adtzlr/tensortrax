import numpy as np
from . import _array as array


class Tensor:
    "A rank-2 tensor."

    def __init__(self, real, dual=None):

        self.real = real
        self.dual = dual
        self.dim = real.shape[0]
        self.shape = real.shape[:2]
        self.dim = self.shape[0]
        self.size = np.product(self.shape)
        self.trax = len(real.shape[2:])

        if self.dual is None:
            self.dual = np.eye(self.size).reshape(*self.shape, *self.shape)

    def __add__(self, B):
        return add(self, B)

    def __sub__(self, B):
        return sub(self, B)

    def __mul__(self, B):
        return mul(self, B)

    def __truediv__(self, B):
        return truediv(self, B)

    def __pow__(self, a):
        return power(self, a)

    __radd__ = __add__
    __rsub__ = __sub__
    __rmul__ = __mul__
    __rtruediv__ = __truediv__


class Constant(Tensor):
    def __init__(self, real):
        super().__init__(real=real, dual=0)


def transpose(A):
    return Tensor(
        real=array.transpose(A.real),
        dual=array.transpose_major(A.dual),
    )


def add(A, B):
    real = A.real + B.real
    dual = A.dual
    if isinstance(B, Tensor):
        dual += B.dual
    return Tensor(real=real, dual=dual)


def sub(A, B):
    real = A.real - B.real
    dual = A.dual
    if isinstance(B, Tensor):
        dual -= B.dual
    return Tensor(real=real, dual=dual)


def mul(A, B):
    real = array.dya(A.real, B.real)
    if isinstance(B, Tensor):
        dual = array.dya(B.real, A.dual) + array.dya(A.real, B.dual)
    else:
        dual = array.dya(B, A.dual)
    return Tensor(real=real, dual=dual)


def truediv(A, B):
    real = A.real / B.real
    if isinstance(B, Tensor):
        raise NotImplementedError("Divide by Tensor is not supported.")
    else:
        dual = A.dual / B
    return Tensor(real=real, dual=dual)


def power(A, a):
    real = A.real**a
    dual = a * A.real ** (a - 1) * A.dual
    return Tensor(real=real, dual=dual)


def dot(A, B):
    real = array.dot(A.real, B.real)
    dual = np.einsum("ikrs...,kj...->ijrs...", A.dual, B.real) + np.einsum(
        "ik...,kjrs...->ijrs...", A.real, B.dual
    )
    return Tensor(real=real, dual=dual)


def ddot(A, B):
    real = array.ddot(A.real, B.real)
    dual = np.einsum("ijrs...,ij...->rs...", A.dual, B.real) + np.einsum(
        "ijrs...,ij...->rs...", B.dual, A.real
    )
    return Tensor(real=real, dual=dual)


def trace(A):
    real = np.trace(A.real)
    dual = np.einsum("ij...,ij->ij...", np.ones_like(A.real), np.eye(A.dim))
    return Tensor(real=real, dual=dual)


def det(A):
    real = array.det(A.real)
    dual = array.det(A.real) * array.transpose(array.inv(A.real))
    return Tensor(real=real, dual=dual)


def inv(A):
    real = array.inv(A.real)
    dual = -np.einsum("il...,kj...->ijkl...", array.inv(A.real), array.inv(A.real))
    return Tensor(real=real, dual=dual)
