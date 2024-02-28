"""
tensorTRAX: Math on (Hyper-Dual) Tensors with Trailing Axes.
"""

from ._special_tensor import (
    ddot,
    dev,
    from_triu_1d,
    from_triu_2d,
    sym,
    tresca,
    triu_1d,
    von_mises,
)

__all__ = [
    "ddot",
    "dev",
    "from_triu_1d",
    "from_triu_2d",
    "sym",
    "tresca",
    "triu_1d",
    "von_mises",
]
