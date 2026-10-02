"""
tensorTRAX: Math on (Hyper-Dual) Tensors with Trailing Axes.
"""

import numpy as np

from .._tensor import Tensor, Zero, Δ, Δδ, dense, einsum, f, matmul, δ

dot = matmul


def array(object, dtype=None, like=None, shape=None):
    """Create a tensor or an array from another tensor, an array or from a list/tuple of
    tensors or arrays.

    Parameters
    ----------
    object : tensortrax.Tensor, array_like, list or tuple of tensortrax.Tensor or list or tuple of array_like
        The object from which the array is created.
    dtype : data-type or None, optional
        Data-type of the array(s). Default is None.
    like : tensortrax.Tensor or None, optional
        Reference tensor for shape and (number of) trailing axes. Default is None. Only
        considered if ``object`` is not a tensor.
    shape : tuple of int or None, optional
        The shape of the data of the tensor (without shape of trailing axes). If None,
        the shape is taken from ``like``. . Only considered if ``object`` is not a
        tensor.

    Returns
    -------
    tensortrax.Tensor or ndarray
        The return type depends on the type of ``object``.
    """

    if isinstance(object, Tensor):
        object = dense(object)
        return Tensor(
            x=np.array(f(object), dtype=dtype),
            δx=np.array(δ(object), dtype=dtype),
            Δx=np.array(Δ(object), dtype=dtype),
            Δδx=np.array(Δδ(object), dtype=dtype),
            ntrax=object.ntrax,
        )
    elif isinstance(object, list) or isinstance(object, tuple):
        if isinstance(object[0], Tensor):
            object = [dense(o) for o in object]
            return Tensor(
                x=np.array([f(o) for o in object], dtype=dtype),
                δx=np.array([δ(o) for o in object], dtype=dtype),
                Δx=np.array([Δ(o) for o in object], dtype=dtype),
                Δδx=np.array([Δδ(o) for o in object], dtype=dtype),
                ntrax=min([o.ntrax for o in object]),
            )
        else:
            return np.array(object, dtype=dtype)
    else:
        if like is None:
            return np.array(object, dtype=dtype)
        else:
            x = np.array(object, dtype=dtype)
            if shape is None:
                shape = like.shape
            return Tensor(x=x.reshape(*shape, *like.trax), ntrax=like.ntrax)


def trace(A):
    "Return the sum along diagonals of the array."
    return einsum("ii...->...", A)


def transpose(A):
    "Returns an array with axes transposed."
    return einsum("ij...->ji...", A)


def sum(A, axis=0):
    "Sum of array elements over a given axis."
    if isinstance(A, Tensor):
        # map the axis argument to the tensor axes (negative values count from the
        # last tensor axis), the trailing (dual and batch) axes are never summed
        axes = np.arange(len(A.shape))
        if axis is not None:
            axes = axes[np.asarray(axis)]
        axis = tuple(np.atleast_1d(axes).tolist())
        return Tensor(
            x=np.sum(f(A), axis=axis),
            δx=np.sum(δ(A), axis=axis),
            Δx=np.sum(Δ(A), axis=axis),
            Δδx=np.sum(Δδ(A), axis=axis),
            ntrax=A.ntrax,
        )
    else:
        return np.sum(A, axis=axis)


def sign(A):
    "Returns an element-wise indication of the sign of a number."
    if isinstance(A, Tensor):
        return Tensor(
            x=np.sign(f(A)),
            δx=0 * δ(A),
            Δx=0 * Δ(A),
            Δδx=0 * Δδ(A),
            ntrax=A.ntrax,
        )
    else:
        return np.sign(A)


def abs(A):
    "Calculate the absolute value element-wise."
    if isinstance(A, Tensor):
        return Tensor(
            x=np.abs(f(A)),
            δx=np.sign(f(A)) * δ(A),
            Δx=np.sign(f(A)) * Δ(A),
            Δδx=np.sign(f(A)) * Δδ(A),
            ntrax=A.ntrax,
        )
    else:
        return np.abs(A)


def sqrt(A):
    "Return the non-negative square-root of an array, element-wise."
    if isinstance(A, Tensor):
        return A**0.5
    else:
        return np.sqrt(A)


def sin(A):
    "Trigonometric sine, element-wise."
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
    "Cosine element-wise."
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
    "Compute tangent element-wise."
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
    "Hyperbolic sine, element-wise."
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
    "Hyperbolic cosine, element-wise."
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
    "Compute hyperbolic tangent element-wise."
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
    "Calculate the exponential of all elements in the input array."
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
    "Natural logarithm, element-wise."
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
    "Return the base 10 logarithm of the input array, element-wise."
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
    "Return specified diagonals."

    def _diagonal(a):
        # np.diagonal appends the diagonal as last axis -> move it to the front,
        # keep the order of all other (tensor and trailing) axes
        if isinstance(a, Zero):
            return a
        d = np.diagonal(a, offset=offset, axis1=axis1, axis2=axis2)
        return np.moveaxis(d, -1, 0)

    if isinstance(A, Tensor):
        return Tensor(
            x=_diagonal(f(A)),
            δx=_diagonal(δ(A)),
            Δx=_diagonal(Δ(A)),
            Δδx=_diagonal(Δδ(A)),
            ntrax=A.ntrax,
        )
    else:
        return _diagonal(A)


def tile(A, reps):
    "Construct an array by repeating A the number of times given by reps."

    if isinstance(A, Tensor):
        A = dense(A)
        return Tensor(
            x=np.tile(f(A), reps=reps),
            δx=np.tile(δ(A), reps=reps),
            Δx=np.tile(Δ(A), reps=reps),
            Δδx=np.tile(Δδ(A), reps=reps),
            ntrax=A.ntrax,
        )
    else:
        return np.tile(A, reps=reps)


def repeat(a, repeats, axis=None):
    "Repeat elements of an array."

    if isinstance(a, Tensor):
        a = dense(a)
        return Tensor(
            x=np.repeat(f(a), repeats=repeats, axis=axis),
            δx=np.repeat(δ(a), repeats=repeats, axis=axis),
            Δx=np.repeat(Δ(a), repeats=repeats, axis=axis),
            Δδx=np.repeat(Δδ(a), repeats=repeats, axis=axis),
            ntrax=a.ntrax,
        )
    else:
        return np.repeat(a, repeats=repeats, axis=axis)


def hstack(tup):
    "Stack arrays in sequence horizontally (column wise)."

    if isinstance(tup[0], Tensor):
        tup = [dense(A) for A in tup]
        return Tensor(
            x=np.hstack([f(A) for A in tup]),
            δx=np.hstack([δ(A) for A in tup]),
            Δx=np.hstack([Δ(A) for A in tup]),
            Δδx=np.hstack([Δδ(A) for A in tup]),
            ntrax=min([A.ntrax for A in tup]),
        )
    else:
        return np.hstack(tup)


def vstack(tup):
    "Stack arrays in sequence vertically (row wise)."

    if isinstance(tup[0], Tensor):
        tup = [dense(A) for A in tup]
        return Tensor(
            x=np.vstack([f(A) for A in tup]),
            δx=np.vstack([δ(A) for A in tup]),
            Δx=np.vstack([Δ(A) for A in tup]),
            Δδx=np.vstack([Δδ(A) for A in tup]),
            ntrax=min([A.ntrax for A in tup]),
        )
    else:
        return np.vstack(tup)


def stack(arrays, axis=0):
    "Join a sequence of arrays along a new axis."

    if isinstance(arrays[0], Tensor):
        arrays = [dense(A) for A in arrays]
        return Tensor(
            x=np.stack([f(A) for A in arrays], axis=axis),
            δx=np.stack([δ(A) for A in arrays], axis=axis),
            Δx=np.stack([Δ(A) for A in arrays], axis=axis),
            Δδx=np.stack([Δδ(A) for A in arrays], axis=axis),
            ntrax=min([A.ntrax for A in arrays]),
        )
    else:
        return np.stack(arrays, axis=axis)


def concatenate(arrays, axis=0):
    "Join a sequence of arrays along an existing axis."

    if isinstance(arrays[0], Tensor):
        arrays = [dense(A) for A in arrays]
        return Tensor(
            x=np.concatenate([f(A) for A in arrays], axis=axis),
            δx=np.concatenate([δ(A) for A in arrays], axis=axis),
            Δx=np.concatenate([Δ(A) for A in arrays], axis=axis),
            Δδx=np.concatenate([Δδ(A) for A in arrays], axis=axis),
            ntrax=min([A.ntrax for A in arrays]),
        )
    else:
        return np.concatenate(arrays, axis=axis)


def split(ary, indices_or_sections, axis=0):
    "Split an array into multiple sub-arrays as views into ary."

    if isinstance(ary, Tensor):
        ary = dense(ary)
        xs = np.split(f(ary), indices_or_sections=indices_or_sections, axis=axis)
        δxs = np.split(δ(ary), indices_or_sections=indices_or_sections, axis=axis)
        Δxs = np.split(Δ(ary), indices_or_sections=indices_or_sections, axis=axis)
        Δδxs = np.split(Δδ(ary), indices_or_sections=indices_or_sections, axis=axis)
        return [
            Tensor(x, δx, Δx, Δδx, ntrax=ary.ntrax)
            for x, δx, Δx, Δδx in zip(xs, δxs, Δxs, Δδxs)
        ]
    else:
        return np.split(ary, indices_or_sections=indices_or_sections, axis=axis)


def external(x, function, gradient, hessian, indices="ij", *args, **kwargs):
    """Evaluate the Tensor returned by an external scalar-valued function, evaluated at
    a given value `x`, with provided gradient and hessian which operates on the values
    of a tensor and optional arguments. All math methods inside the external
    function/gradient/hessian must handle arbitrary number of elementwise-operating
    trailing axes.
    """

    # pre-evaluate the scalar-valued function along with its gradient and hessian
    if isinstance(x, Tensor):
        func = function(f(x), *args, **kwargs)
        grad = gradient(f(x), *args, **kwargs)
        hess = hessian(f(x), *args, **kwargs)

    def gvp(g, v, ntrax):
        "Evaluate the gradient-vector product."

        ij = indices.lower()

        return einsum(f"{ij}...,{ij}...->...", g, v)

    def hvp(h, v, u, ntrax):
        "Evaluate the hessian-vectors product."

        ij = indices.lower()
        kl = indices.upper()

        return einsum(f"{ij}{kl}...,{ij}...,{kl}...->...", h, v, u)

    if isinstance(x, Tensor):
        return Tensor(
            x=func,
            δx=gvp(grad, δ(x), x.ntrax),
            Δx=gvp(grad, Δ(x), x.ntrax),
            Δδx=hvp(hess, δ(x), Δ(x), x.ntrax) + gvp(grad, Δδ(x), x.ntrax),
            ntrax=x.ntrax,
        )
    else:
        return function(x, *args, **kwargs)


def if_else(cond, true, false):
    "Mask-based Condition for arrays and tensors."

    mask = np.asarray(cond)

    def where(a, b):
        "Select items of a or b by the mask, structural zeros remain zero."
        if isinstance(a, Zero) and isinstance(b, Zero):
            return a
        a = 0 if isinstance(a, Zero) else a
        b = 0 if isinstance(b, Zero) else b
        return np.where(mask, a, b)

    if isinstance(true, np.ndarray) and isinstance(false, np.ndarray):
        return where(true, false)

    elif isinstance(true, Tensor) and isinstance(false, Tensor):
        return Tensor(
            x=where(f(true), f(false)),
            δx=where(δ(true), δ(false)),
            Δx=where(Δ(true), Δ(false)),
            Δδx=where(Δδ(true), Δδ(false)),
            ntrax=true.ntrax,
        )

    else:
        raise NotImplementedError(
            "`true` and `false` must be both arrays or both tensors."
        )


def maximum(x1, x2):
    "Element-wise maximum of array elements."

    if isinstance(x1, Tensor):
        return if_else(x1 > x2, x1, x2)
    else:
        return np.maximum(x1, x2)


def minimum(x1, x2):
    "Element-wise minimum of array elements."

    if isinstance(x1, Tensor):
        return if_else(x1 < x2, x1, x2)
    else:
        return np.minimum(x1, x2)
