"""
tensorTRAX: Math on (Hyper-Dual) Tensors with Trailing Axes.
"""

from .._tensor import broadcast_to, dual2real, ravel, reshape, squeeze
from . import _math_array as base
from . import linalg, special
from ._math_tensor import (
    abs,
    array,
    concatenate,
    cos,
    cosh,
    diagonal,
    dot,
    einsum,
    exp,
    external,
    hstack,
    if_else,
    log,
    log10,
    matmul,
    maximum,
    minimum,
    repeat,
    sign,
    sin,
    sinh,
    split,
    sqrt,
    stack,
    sum,
    tan,
    tanh,
    tile,
    trace,
    transpose,
    vstack,
)

__all__ = [
    "linalg",
    "base",
    "special",
    "abs",
    "array",
    "broadcast_to",
    "cos",
    "cosh",
    "concatenate",
    "diagonal",
    "dot",
    "dual2real",
    "einsum",
    "exp",
    "external",
    "hstack",
    "log",
    "log10",
    "matmul",
    "maximum",
    "minimum",
    "ravel",
    "repeat",
    "reshape",
    "sign",
    "sin",
    "sinh",
    "split",
    "sqrt",
    "squeeze",
    "stack",
    "sum",
    "tan",
    "tanh",
    "tile",
    "trace",
    "transpose",
    "vstack",
    "if_else",
]
