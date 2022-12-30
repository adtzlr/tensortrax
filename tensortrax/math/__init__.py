r"""
 _                            
| |                          ████████╗██████╗  █████╗ ██╗  ██╗
| |_ ___ _ __  ___  ___  _ __╚══██╔══╝██╔══██╗██╔══██╗╚██╗██╔╝
| __/ _ \ '_ \/ __|/ _ \| '__|  ██║   ██████╔╝███████║ ╚███╔╝ 
| ||  __/ | | \__ \ (_) | |     ██║   ██╔══██╗██╔══██║ ██╔██╗ 
 \__\___|_| |_|___/\___/|_|     ██║   ██║  ██║██║  ██║██╔╝ ██╗
                                ╚═╝   ╚═╝  ╚═╝╚═╝  ╚═╝╚═╝  ╚═╝  
"""

from . import _linalg as linalg
from . import _math_array as array
from . import _special as special
from ._math_tensor import (
    cos,
    cosh,
    diagonal,
    dot,
    einsum,
    exp,
    log,
    log10,
    matmul,
    ravel,
    reshape,
    sin,
    sinh,
    sqrt,
    sum,
    tan,
    tanh,
    trace,
    transpose,
)
