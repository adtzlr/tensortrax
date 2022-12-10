import tensortrax as tr
import tensortrax.math as tm
import numpy as np
import pytest


def test_math():

    F = np.eye(3) + np.arange(9).reshape(3, 3) / 10
    T = tr.Tensor(F)
    print(T)

    C = F.T @ F

    assert np.allclose(tr.f(T.T() @ F), C)
    assert np.allclose(tr.f(F.T @ T), C)
    assert np.allclose(tr.f(T.T() @ T), C)

    assert T[0].shape == (3,)

    assert isinstance(T * F, tr.Tensor)
    assert isinstance(F * T, tr.Tensor)
    assert isinstance(T * T, tr.Tensor)

    assert isinstance(T / F, tr.Tensor)
    assert isinstance(F / T, tr.Tensor)
    with pytest.raises(NotImplementedError):
        T / T

    assert isinstance(T + F, tr.Tensor)
    assert isinstance(F + T, tr.Tensor)
    assert isinstance(T + T, tr.Tensor)

    assert isinstance(T - F, tr.Tensor)
    assert isinstance(F - T, tr.Tensor)
    assert isinstance(T - T, tr.Tensor)

    assert np.allclose((-T).x, -F)

    F = np.eye(3) + np.arange(1, 10).reshape(3, 3) / 10
    T = tr.Tensor(F)

    assert np.allclose(tm.linalg.det(F), tm.linalg.det(T).x)
    assert np.allclose(tm.linalg.inv(F), tm.linalg.inv(T).x)

    tm.linalg._det(F[:2, :2])
    tm.linalg._det(F[:1, :1])
    tm.linalg._inv(F[:2, :2])

    for fun in [
        tm.sin,
        tm.cos,
        tm.tan,
        tm.sinh,
        tm.cosh,
        tm.tanh,
        tm.sqrt,
        tm.exp,
        tm.log,
        tm.log10,
        tm.diagonal,
        tm.ravel,
    ]:
        assert np.allclose(fun(F), fun(T).x)

    for fun in [tm.linalg.det]:
        assert np.allclose(fun(F), fun(T).x)

    assert tm.linalg.eigvalsh(T).shape == (3,)

    assert tm.array.cross(F, F).shape == F.shape
    assert tm.array.eye(F).shape == F.shape

    tm.einsum("ij...,kl...->ijkl...", F, F)
    tm.einsum("ij...,kl...->ijkl...", F, T)
    tm.einsum("ij...,kl...->ijkl...", T, F)
    tm.einsum("ij...,kl...->ijkl...", T, T)

    tm.einsum("ij...,kl...,mn...->ijklmn...", F, F, F)
    tm.einsum("ij...,kl...,mn...->ijklmn...", F, F, T)
    tm.einsum("ij...,kl...,mn...->ijklmn...", F, T, F)
    tm.einsum("ij...,kl...,mn...->ijklmn...", F, T, T)
    tm.einsum("ij...,kl...,mn...->ijklmn...", T, F, F)
    tm.einsum("ij...,kl...,mn...->ijklmn...", T, F, T)
    tm.einsum("ij...,kl...,mn...->ijklmn...", T, T, F)
    tm.einsum("ij...,kl...,mn...->ijklmn...", T, T, T)

    with pytest.raises(NotImplementedError):
        tm.einsum("ij...,kl...,mn...,pq...->ijklmnpq...", T, T, T, T)

    T.ravel()
    T[0] = F[0]
    T[:, 0] = F[:, 0]
    T[:, 0] = T[:, 0]


if __name__ == "__main__":
    test_math()
