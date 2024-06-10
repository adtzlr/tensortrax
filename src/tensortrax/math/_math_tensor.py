"""
tensorTRAX: Math on (Hyper-Dual) Tensors with Trailing Axes.
"""
import numpy as np

from .._tensor import Tensor, Δ, Δδ, broadcast_to, einsum, f, matmul, δ

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
        return Tensor(
            x=np.array(f(object), dtype=dtype),
            δx=np.array(δ(object), dtype=dtype),
            Δx=np.array(Δ(object), dtype=dtype),
            Δδx=np.array(Δδ(object), dtype=dtype),
            ntrax=object.ntrax,
        )
    elif isinstance(object, list) or isinstance(object, tuple):
        if isinstance(object[0], Tensor):
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


def tile(A, reps):
    "Construct an array by repeating A the number of times given by reps."

    if isinstance(A, Tensor):
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
    out = true.copy()

    if isinstance(true, np.ndarray) and isinstance(false, np.ndarray):
        out = true.copy()
        out[..., mask] = true[..., mask]
        out[..., ~mask] = false[..., ~mask]

    elif isinstance(true, Tensor) and isinstance(false, Tensor):
        shape = np.maximum.reduce(
            [
                true.x.shape,
                true.δx.shape,
                true.Δx.shape,
                true.Δδx.shape,
                false.x.shape,
                false.δx.shape,
                false.Δx.shape,
                false.Δδx.shape,
            ]
        )

        out = broadcast_to(true, shape=shape).copy()

        mask = np.broadcast_to(mask, shape)
        out[..., ~mask] = broadcast_to(false, shape=shape)[..., ~mask]

    else:
        raise NotImplementedError(
            "`true` and `false` must be both arrays or both tensors."
        )

    return out


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
