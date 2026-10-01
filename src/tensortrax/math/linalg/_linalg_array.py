"""
tensorTRAX: Math on (Hyper-Dual) Tensors with Trailing Axes.
"""

import numpy as np


def det(A):
    "Determinant of an Array."
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
    else:
        detA = np.linalg.det(A.T).T
    return detA


def adj(A):
    "Adjugate (transpose of the cofactor matrix) of an Array."

    detAinvA = np.zeros_like(A)

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

    elif A.shape[0] == 1:
        detAinvA[0, 0] = 1

    else:
        detAinvA = det(A) * np.linalg.inv(A.T).T

    return detAinvA


def inv(A):
    "Inverse of an Array."
    return adj(A) / det(A)


def adj_variation(A, dA):
    "Variation of the adjugate of an Array in direction dA."

    if A.shape[0] == 3:

        def minor(i, j, k, l):
            "Variation of A[i, j] * A[k, l]."
            return A[i, j] * dA[k, l] + dA[i, j] * A[k, l]

        dadjA = np.zeros(np.broadcast_shapes(A.shape, dA.shape))

        dadjA[0, 0] = minor(1, 1, 2, 2) - minor(1, 2, 2, 1)
        dadjA[1, 1] = minor(0, 0, 2, 2) - minor(0, 2, 2, 0)
        dadjA[2, 2] = minor(0, 0, 1, 1) - minor(0, 1, 1, 0)

        dadjA[0, 1] = minor(0, 2, 2, 1) - minor(0, 1, 2, 2)
        dadjA[0, 2] = minor(0, 1, 1, 2) - minor(0, 2, 1, 1)
        dadjA[1, 2] = minor(0, 2, 1, 0) - minor(0, 0, 1, 2)

        dadjA[1, 0] = minor(1, 2, 2, 0) - minor(1, 0, 2, 2)
        dadjA[2, 0] = minor(1, 0, 2, 1) - minor(1, 1, 2, 0)
        dadjA[2, 1] = minor(0, 1, 2, 0) - minor(0, 0, 2, 1)

    elif A.shape[0] == 2:
        dadjA = adj(dA)  # the adjugate of a 2x2 matrix is linear

    elif A.shape[0] == 1:
        dadjA = np.zeros_like(dA)  # the adjugate of a 1x1 matrix is constant

    else:
        # adj(A) = det(A) inv(A), requires a regular matrix
        invA = inv(A)
        invAdA = np.einsum("ik...,kj...->ij...", invA, dA)
        dadjA = det(A) * (
            np.einsum("ii...->...", invAdA) * invA
            - np.einsum("ik...,kj...->ij...", invAdA, invA)
        )

    return dadjA


def pinv(A, hermitian=False):
    "Pseudo-inverse of an Array."

    return np.linalg.pinv(A.T, hermitian=hermitian).T
