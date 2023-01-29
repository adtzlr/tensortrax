"""
tensorTRAX: Math on (Hyper-Dual) Tensors with Trailing Axes.
"""

from . import math
from .__about__ import __version__
from ._evaluate import (
    function,
    gradient,
    gradient_vector_product,
    hessian,
    hessian_vector_product,
    hessian_vectors_product,
    jacobian,
    take,
)
from ._helpers import Δ, Δδ, f, δ
from ._tensor import Tensor

__all__ = [
    "__version__",
]
