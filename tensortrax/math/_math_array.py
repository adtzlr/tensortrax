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


def eye(A):
    "Identity (Eye) of a Tensor."
    B = np.zeros_like(A)
    B[np.diag_indices(B.shape[0])] = 1
    return B


def cross(a, b):
    "Cross product of two vectors a and b."
    return np.einsum(
        "...i->i...", np.cross(np.einsum("i...->...i", a), np.einsum("i...->...i", b))
    )
