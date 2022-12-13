r"""
 _                            
| |                          ████████╗██████╗  █████╗ ██╗  ██╗
| |_ ___ _ __  ___  ___  _ __╚══██╔══╝██╔══██╗██╔══██╗╚██╗██╔╝
| __/ _ \ '_ \/ __|/ _ \| '__|  ██║   ██████╔╝███████║ ╚███╔╝ 
| ||  __/ | | \__ \ (_) | |     ██║   ██╔══██╗██╔══██║ ██╔██╗ 
 \__\___|_| |_|___/\___/|_|     ██║   ██║  ██║██║  ██║██╔╝ ██╗
                                ╚═╝   ╚═╝  ╚═╝╚═╝  ╚═╝╚═╝  ╚═╝  
"""


# from ..._tensor import Tensor, einsum, matmul, f, δ, Δ, Δδ
from .._math_tensor import trace, ddot, sqrt
from .. import _math_array as array
from .._linalg import _linalg_tensor as linalg


def dev(A):
    "Deviatoric Part of a Tensor."
    dim = A.shape[0]
    return A - trace(A) / dim * array.eye(A)


def tresca(A):
    "Tresca Invariant."
    λ = linalg.eigvalsh(A)
    return (λ[-1] - λ[0]) / 2


def von_mises(A):
    "Von Mises Invariant."
    a = dev(A)
    return sqrt(3 / 2 * ddot(a, a))
