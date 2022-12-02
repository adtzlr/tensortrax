r"""
 _                            
| |                          ████████╗██████╗  █████╗ ██╗  ██╗
| |_ ___ _ __  ___  ___  _ __╚══██╔══╝██╔══██╗██╔══██╗╚██╗██╔╝
| __/ _ \ '_ \/ __|/ _ \| '__|  ██║   ██████╔╝███████║ ╚███╔╝ 
| ||  __/ | | \__ \ (_) | |     ██║   ██╔══██╗██╔══██║ ██╔██╗ 
 \__\___|_| |_|___/\___/|_|     ██║   ██║  ██║██║  ██║██╔╝ ██╗
                                ╚═╝   ╚═╝  ╚═╝╚═╝  ╚═╝╚═╝  ╚═╝  
"""

from ._array import det as _det, inv as _inv, eye as _eye
from ._tensor import (
    sin,
    cos,
    tan,
    tanh,
    dot,
    ddot,
    trace,
    transpose,
    sum,
    sqrt,
    det,
    eigvalsh,
    einsum,
    matmul,
    function,
    gradient,
    hessian,
    gradient_vector_product,
    hessian_vector_product,
)
