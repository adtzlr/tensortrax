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


def det(A):
    "Determinant of a 2x2 or 3x3 Array."
    if A.shape[0] == 3:
        detA = (
            A[0, 0] * A[1, 1] * A[2, 2]
            + A[0, 1] * A[1, 2] * A[2, 0]
            + A[0, 2] * A[1, 0] * A[2, 1]
            - A[2, 0] * A[1, 1] * A[0, 2]
            - A[2, 1] * A[1, 2] * A[0, 0]
            - A[2, 2] * A[1, 0] * A[0, 1]
        )
    elif A.shape[0] == 2:
        detA = A[0, 0] * A[1, 1] - A[0, 1] * A[1, 0]
    elif A.shape[0] == 1:
        detA = A[0, 0]
    return detA


def inv(A):
    "Inverse of a 2x2 or 3x3 Array."

    detAinvA = np.zeros_like(A)
    detA = det(A)

    if A.shape[0] == 3:

        detAinvA[0, 0] = -A[1, 2] * A[2, 1] + A[1, 1] * A[2, 2]
        detAinvA[1, 1] = -A[0, 2] * A[2, 0] + A[0, 0] * A[2, 2]
        detAinvA[2, 2] = -A[0, 1] * A[1, 0] + A[0, 0] * A[1, 1]

        detAinvA[0, 1] = A[0, 2] * A[2, 1] - A[0, 1] * A[2, 2]
        detAinvA[0, 2] = -A[0, 2] * A[1, 1] + A[0, 1] * A[1, 2]
        detAinvA[1, 2] = A[0, 2] * A[1, 0] - A[0, 0] * A[1, 2]

        detAinvA[1, 0] = A[1, 2] * A[2, 0] - A[1, 0] * A[2, 2]
        detAinvA[2, 0] = -A[1, 1] * A[2, 0] + A[1, 0] * A[2, 1]
        detAinvA[2, 1] = A[0, 1] * A[2, 0] - A[0, 0] * A[2, 1]

    elif A.shape[0] == 2:
        detAinvA[0, 0] = A[1, 1]
        detAinvA[0, 1] = -A[0, 1]
        detAinvA[1, 0] = -A[1, 0]
        detAinvA[1, 1] = A[0, 0]

    return detAinvA / detA
