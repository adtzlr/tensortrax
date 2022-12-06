"""
tensorTRAX: Math on (Hyper-Dual) Tensors with Trailing Axes.
"""

from ._tensor import Tensor
from ._helpers import f, δ, Δ, Δδ
from ._evaluate import (
    function,
    gradient,
    hessian,
    gradient_vector_product,
    hessian_vector_product,
)
from . import math
