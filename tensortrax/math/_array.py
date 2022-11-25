import numpy as np


def transpose(A):
    return np.einsum("ij...->ji...", A)


def transpose_major(A):
    return np.einsum("ijrs...->rsij...", A)


def dya(A, B, trax=2):
    a = "abcdef"[: len(A.shape[:-trax])]
    b = "rstuvw"[: len(B.shape[:-trax])]
    return np.einsum(f"{a}...,{b}...->{a}{b}...", A, B)


def dot(A, B):
    return np.einsum("ik...,kj...->ij...", A, B)


def ddot(A, B):
    return np.einsum("ij...,ij...->...", A, B)


def det(A):
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
