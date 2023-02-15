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

from ._helpers import Δ, Δδ, f, δ


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

    def __init__(self, x, δx=None, Δx=None, Δδx=None, ntrax=0, ndual=0):
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
            Number of trailing axes (default is 0).
        ndual : int, optional
            Number of axes containing dual data (default is 0).

        """

        self.x = np.asarray(x)

        self.ntrax = ntrax
        self.ndual = ndual

        self.shape = self.x.shape[: len(self.x.shape) - ntrax]
        self.trax = self.x.shape[len(self.x.shape) - ntrax :]
        self.size = np.product(self.shape, dtype=int)

        self.δx = self._init_and_reshape(δx)
        self.Δx = self._init_and_reshape(Δx)
        self.Δδx = self._init_and_reshape(Δδx)

    def _add(self, ndual=1):
        "Add additional trailing axes for dual values."

        # reset the tensor
        self.reset()

        # add new dual-related axes to the shape of the data
        ones = np.ones(ndual, dtype=int)
        x = self.x.reshape(*self.shape, *ones, *self.trax)

        # re-init the tensor
        self.__init__(
            x,
            δx=self.δx,
            Δx=self.Δx,
            Δδx=self.Δδx,
            ntrax=self.ntrax + ndual,
            ndual=ndual,
        )

    def reset(self):
        "Reset the Tensor (set all dual values to zero)."

        # reset the shape of the data (delete axes related to dual data)
        x = self.x.reshape(*self.shape, *self.trax[self.ndual :])

        # re-init the tensor
        self.__init__(x, ntrax=self.ntrax - self.ndual)

    def init(self, gradient=False, hessian=False, sym=False, δx=None, Δx=None):
        """Re-Initialize tensor with dual values to keep track of the
        hessian and/or the gradient."""

        if gradient and not hessian:
            # add additional element-wise acting axes for dual values
            self._add(ndual=len(self.shape))

            # create the dual values
            if δx is None:
                δx = np.eye(self.size).reshape(*self.shape, *self.shape)
                Δx = δx.copy()

        elif hessian:
            # add additional trailing axes for dual values
            self._add(ndual=2 * len(self.shape))

            # create the dual values
            ones = np.ones(len(self.shape), dtype=int)

            if δx is None or δx is False:
                shape = (*self.shape, *self.shape, *ones)
                if len(shape) == 0:
                    shape = (1,)
                if δx is False:
                    δx = np.zeros(self.size**2).reshape(shape)
                else:
                    δx = np.eye(self.size).reshape(shape)
            else:
                δx = δx.reshape(*self.shape, *self.trax)

            if Δx is None or Δx is False:
                shape = (*self.shape, *ones, *self.shape)
                if len(shape) == 0:
                    shape = (1,)
                if Δx is False:
                    Δx = np.zeros(self.size**2).reshape(shape)
                else:
                    Δx = np.eye(self.size).reshape(shape)
            else:
                Δx = Δx.reshape(*self.shape, *self.trax)

        if gradient or hessian:

            if sym:
                idx_off_diag = {1: None, 3: [1], 6: [1, 2, 4]}[self.size]
                δx[idx_off_diag] /= 2
                Δx[idx_off_diag] /= 2

            # re-init the tensor
            self.__init__(self.x, δx=δx, Δx=Δx, ntrax=self.ntrax, ndual=self.ndual)

    def _init_and_reshape(self, value):
        if value is None:
            value = np.zeros(self.shape)
        else:
            value = np.asarray(value)
        if len(value.shape) != len(self.x.shape):
            newshape = (
                *value.shape,
                *np.ones(len(self.x.shape) - len(value.shape), dtype=int),
            )
            value = value.reshape(newshape)
        return value

    def __neg__(self):
        return Tensor(
            x=-self.x, δx=-self.δx, Δx=-self.Δx, Δδx=-self.Δδx, ntrax=self.ntrax
        )

    def __add__(self, B):
        A = self
        if isinstance(B, Tensor):
            x = f(A) + f(B)
            δx = δ(A) + δ(B)
            Δx = Δ(A) + Δ(B)
            Δδx = Δδ(A) + Δδ(B)
            ntrax = min(A.ntrax, B.ntrax)
        else:
            x = f(A) + B
            δx = δ(A)
            Δx = Δ(A)
            Δδx = Δδ(A)
            ntrax = A.ntrax
        return Tensor(x=x, δx=δx, Δx=Δx, Δδx=Δδx, ntrax=ntrax)

    def __sub__(self, B):
        A = self
        if isinstance(B, Tensor):
            x = f(A) - f(B)
            δx = δ(A) - δ(B)
            Δx = Δ(A) - Δ(B)
            Δδx = Δδ(A) - Δδ(B)
            ntrax = min(A.ntrax, B.ntrax)
        else:
            x = f(A) - B
            δx = δ(A)
            Δx = Δ(A)
            Δδx = Δδ(A)
            ntrax = A.ntrax
        return Tensor(x=x, δx=δx, Δx=Δx, Δδx=Δδx, ntrax=ntrax)

    def __rsub__(self, B):
        return -self.__sub__(B)

    def __mul__(self, B):
        A = self
        if isinstance(B, Tensor):
            x = f(A) * f(B)
            δx = δ(A) * f(B) + f(A) * δ(B)
            Δx = Δ(A) * f(B) + f(A) * Δ(B)
            Δδx = Δ(A) * δ(B) + δ(A) * Δ(B) + Δδ(A) * f(B) + f(A) * Δδ(B)
            ntrax = min(A.ntrax, B.ntrax)
        else:
            x = f(A) * B
            δx = δ(A) * B
            Δx = Δ(A) * B
            Δδx = Δδ(A) * B
            ntrax = A.ntrax
        return Tensor(x=x, δx=δx, Δx=Δx, Δδx=Δδx, ntrax=ntrax)

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
        Δx = p * f(A) ** (p - 1) * Δ(A)
        Δδx = p * f(A) ** (p - 1) * Δδ(A) + p * (p - 1) * f(A) ** (p - 2) * δ(A) * Δ(A)
        return Tensor(x=x, δx=δx, Δx=Δx, Δδx=Δδx, ntrax=A.ntrax)

    def __gt__(self, B):
        A = self
        return f(A) > (f(B) if isinstance(B, Tensor) else B)

    def __lt__(self, B):
        A = self
        return f(A) < (f(B) if isinstance(B, Tensor) else B)

    def __ge__(self, B):
        A = self
        return f(A) >= (f(B) if isinstance(B, Tensor) else B)

    def __le__(self, B):
        A = self
        return f(A) <= (f(B) if isinstance(B, Tensor) else B)

    def __eq__(self, B):
        A = self
        return f(A) == (f(B) if isinstance(B, Tensor) else B)

    def __matmul__(self, B):
        return matmul(self, B)

    def __rmatmul__(self, B):
        return matmul(B, self)

    def __getitem__(self, key):
        x = f(self)[key]
        Δx = Δ(self)[key]
        δx = δ(self)[key]
        Δδx = Δδ(self)[key]
        return Tensor(x=x, δx=δx, Δx=Δx, Δδx=Δδx, ntrax=self.ntrax)

    def __setitem__(self, key, value):
        if isinstance(value, Tensor):
            shape = (*self.shape, *self.trax)
            self.x[key] = f(value)
            if self.δx[key].shape != δ(value).shape:
                self.δx = broadcast_to(self.δx, shape).copy()
            if self.Δx[key].shape != Δ(value).shape:
                self.Δx = broadcast_to(self.Δx, shape).copy()
            if self.Δδx[key].shape != Δδ(value).shape:
                self.Δδx = broadcast_to(self.Δδx, shape).copy()
            self.δx[key] = δ(value)
            self.Δx[key] = Δ(value)
            self.Δδx[key] = Δδ(value)
        else:
            self.x[key] = value
            self.δx[key].fill(0)
            self.Δx[key].fill(0)
            self.Δδx[key].fill(0)

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

    def squeeze(self, axis=None):
        return squeeze(self, axis=axis)

    def dual2real(self, like):
        return dual2real(self, like=like)

    __radd__ = __add__
    __rmul__ = __mul__
    __array_ufunc__ = None


def broadcast_to(A, shape):
    "Broadcast Array or Tensor to a new shape."

    _broadcast_to = lambda A: np.broadcast_to(A, shape=shape)

    if isinstance(A, Tensor):
        return Tensor(
            x=_broadcast_to(f(A)),
            δx=_broadcast_to(δ(A)),
            Δx=_broadcast_to(Δ(A)),
            Δδx=_broadcast_to(Δδ(A)),
            ntrax=A.ntrax,
        )
    else:
        return _broadcast_to(A)


def dual2real(A, like=None):
    """Return a new Tensor with old-dual data as new-real values,
    with `ntrax` derived by `like`."""

    ndual = like.ndual - len(like.shape)
    ntrax = A.ntrax - ndual

    return Tensor(x=A.δx, δx=A.Δδx, Δx=A.Δδx, ndual=min(ndual, ntrax), ntrax=ntrax)


def ravel(A, order="C"):
    if isinstance(A, Tensor):
        δtrax = δ(A).shape[len(A.shape) :]
        Δtrax = Δ(A).shape[len(A.shape) :]
        Δδtrax = Δδ(A).shape[len(A.shape) :]
        return Tensor(
            x=f(A).reshape(A.size, *A.trax, order=order),
            δx=δ(A).reshape(A.size, *δtrax, order=order),
            Δx=Δ(A).reshape(A.size, *Δtrax, order=order),
            Δδx=Δδ(A).reshape(A.size, *Δδtrax, order=order),
            ntrax=A.ntrax,
        )
    else:
        return np.ravel(A, order=order)


def squeeze(A, axis=None):
    if isinstance(A, Tensor):
        if axis is None:
            if 1 in A.shape:
                axis = tuple(np.arange(len(A.shape))[np.array(A.shape) == 1])
            else:
                axis = ()
        return Tensor(
            x=np.squeeze(f(A), axis=axis),
            δx=np.squeeze(δ(A), axis=axis),
            Δx=np.squeeze(Δ(A), axis=axis),
            Δδx=np.squeeze(Δδ(A), axis=axis),
            ntrax=A.ntrax,
        )
    else:
        return np.squeeze(A, axis=axis)


def reshape(A, newshape, order="C"):
    if isinstance(A, Tensor):
        δtrax = δ(A).shape[len(A.shape) :]
        Δtrax = Δ(A).shape[len(A.shape) :]
        Δδtrax = Δδ(A).shape[len(A.shape) :]
        return Tensor(
            x=f(A).reshape(*newshape, *A.trax, order=order),
            δx=δ(A).reshape(*newshape, *δtrax, order=order),
            Δx=Δ(A).reshape(*newshape, *Δtrax, order=order),
            Δδx=Δδ(A).reshape(*newshape, *Δδtrax, order=order),
            ntrax=A.ntrax,
        )
    else:
        return np.reshape(A, newshape=newshape, order=order)


def einsum3(subscripts, *operands):
    "Einsum with three operands."
    A, B, C = operands
    _einsum = lambda *operands: np.einsum(subscripts, *operands)

    if isinstance(A, Tensor) and isinstance(B, Tensor) and isinstance(C, Tensor):
        x = _einsum(f(A), f(B), f(C))
        δx = (
            _einsum(δ(A), f(B), f(C))
            + _einsum(f(A), δ(B), f(C))
            + _einsum(f(A), f(B), δ(C))
        )
        Δx = (
            _einsum(Δ(A), f(B), f(C))
            + _einsum(f(A), Δ(B), f(C))
            + _einsum(f(A), f(B), Δ(C))
        )
        Δδx = (
            _einsum(Δδ(A), f(B), f(C))
            + _einsum(f(A), Δδ(B), f(C))
            + _einsum(f(A), f(B), Δδ(C))
            + _einsum(δ(A), Δ(B), f(C))
            + _einsum(Δ(A), δ(B), f(C))
            + _einsum(δ(A), f(B), Δ(C))
            + _einsum(Δ(A), f(B), δ(C))
            + _einsum(f(A), δ(B), Δ(C))
            + _einsum(f(A), Δ(B), δ(C))
        )
        ntrax = min(A.ntrax, B.ntrax, C.ntrax)
    elif (
        isinstance(A, Tensor)
        and not isinstance(B, Tensor)
        and not isinstance(C, Tensor)
    ):
        x = _einsum(f(A), B, C)
        δx = _einsum(δ(A), B, C)
        Δx = _einsum(Δ(A), B, C)
        Δδx = _einsum(Δδ(A), B, C)
        ntrax = A.ntrax
    elif (
        not isinstance(A, Tensor)
        and isinstance(B, Tensor)
        and not isinstance(C, Tensor)
    ):
        x = _einsum(A, f(B), C)
        δx = _einsum(A, δ(B), C)
        Δx = _einsum(A, Δ(B), C)
        Δδx = _einsum(A, Δδ(B), C)
        ntrax = B.ntrax
    elif (
        not isinstance(A, Tensor)
        and not isinstance(B, Tensor)
        and isinstance(C, Tensor)
    ):
        x = _einsum(A, B, f(C))
        δx = _einsum(A, B, δ(C))
        Δx = _einsum(A, B, Δ(C))
        Δδx = _einsum(A, B, Δδ(C))
        ntrax = C.ntrax
    elif isinstance(A, Tensor) and isinstance(B, Tensor) and not isinstance(C, Tensor):
        x = _einsum(f(A), f(B), C)
        δx = _einsum(δ(A), f(B), C) + _einsum(f(A), δ(B), C)
        Δx = _einsum(Δ(A), f(B), C) + _einsum(f(A), Δ(B), C)
        Δδx = (
            _einsum(Δδ(A), f(B), C)
            + _einsum(f(A), Δδ(B), C)
            + _einsum(δ(A), Δ(B), C)
            + _einsum(Δ(A), δ(B), C)
        )
        ntrax = min(A.ntrax, B.ntrax)
    elif isinstance(A, Tensor) and not isinstance(B, Tensor) and isinstance(C, Tensor):
        x = _einsum(f(A), B, f(C))
        δx = _einsum(δ(A), B, f(C)) + _einsum(f(A), B, δ(C))
        Δx = _einsum(Δ(A), B, f(C)) + _einsum(f(A), B, Δ(C))
        Δδx = (
            _einsum(Δδ(A), B, f(C))
            + _einsum(f(A), B, Δδ(C))
            + _einsum(δ(A), B, Δ(C))
            + _einsum(Δ(A), B, δ(C))
        )
        ntrax = min(A.ntrax, C.ntrax)
    elif not isinstance(A, Tensor) and isinstance(B, Tensor) and isinstance(C, Tensor):
        x = _einsum(A, f(B), f(C))
        δx = _einsum(A, δ(B), f(C)) + _einsum(A, f(B), δ(C))
        Δx = _einsum(A, Δ(B), f(C)) + _einsum(A, f(B), Δ(C))
        Δδx = (
            _einsum(A, Δδ(B), f(C))
            + _einsum(A, f(B), Δδ(C))
            + _einsum(A, δ(B), Δ(C))
            + _einsum(A, Δ(B), δ(C))
        )
        ntrax = min(B.ntrax, C.ntrax)
    else:
        return _einsum(*operands)

    return Tensor(x=x, δx=δx, Δx=Δx, Δδx=Δδx, ntrax=ntrax)


def einsum2(subscripts, *operands):
    "Einsum with two operands."
    A, B = operands
    _einsum = lambda *operands: np.einsum(subscripts, *operands)

    if isinstance(A, Tensor) and isinstance(B, Tensor):
        x = _einsum(f(A), f(B))
        δx = _einsum(δ(A), f(B)) + _einsum(f(A), δ(B))
        Δx = _einsum(Δ(A), f(B)) + _einsum(f(A), Δ(B))
        Δδx = (
            _einsum(Δδ(A), f(B))
            + _einsum(f(A), Δδ(B))
            + _einsum(δ(A), Δ(B))
            + _einsum(Δ(A), δ(B))
        )
        ntrax = min(A.ntrax, B.ntrax)
    elif isinstance(A, Tensor) and not isinstance(B, Tensor):
        x = _einsum(f(A), B)
        δx = _einsum(δ(A), B)
        Δx = _einsum(Δ(A), B)
        Δδx = _einsum(Δδ(A), B)
        ntrax = A.ntrax
    elif not isinstance(A, Tensor) and isinstance(B, Tensor):
        x = _einsum(A, f(B))
        δx = _einsum(A, δ(B))
        Δx = _einsum(A, Δ(B))
        Δδx = _einsum(A, Δδ(B))
        ntrax = B.ntrax
    else:
        return _einsum(*operands)

    return Tensor(x=x, δx=δx, Δx=Δx, Δδx=Δδx, ntrax=ntrax)


def einsum1(subscripts, *operands):
    "Einsum with one operand."
    A = operands[0]
    _einsum = lambda *operands: np.einsum(subscripts, *operands)
    if isinstance(A, Tensor):
        x = _einsum(f(A))
        δx = _einsum(δ(A))
        Δx = _einsum(Δ(A))
        Δδx = _einsum(Δδ(A))
        return Tensor(x=x, δx=δx, Δx=Δx, Δδx=Δδx, ntrax=A.ntrax)
    else:
        return _einsum(*operands)


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


def transpose(A):
    ij = "abcdefghijklmnopqrstuvwxyz"[: len(A.shape)]
    ji = ij[::-1]
    return einsum(f"{ij}...->{ji}...", A)


def matmul(A, B):
    ik = "ik"[2 - len(A.shape) :]
    kj = "kj"[: len(B.shape)]
    ij = (ik + kj).replace("k", "")
    return einsum(f"{ik}...,{kj}...->{ij}...", A, B)
