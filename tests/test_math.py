import numpy as np
import pytest

import tensortrax as tr
import tensortrax.math as tm


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
    assert isinstance(T / T, tr.Tensor)

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
    tm.linalg._inv(F[:1, :1])

    G = np.eye(4) + np.arange(1, 17).reshape(4, 4) / 10
    tm.linalg._det(G)
    tm.linalg._inv(G)

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
        tm.abs,
        tm.sign,
    ]:
        assert np.allclose(fun(F), fun(T).x)

    C = F.T @ F
    V = tr.Tensor(C)

    for fun in [tm.linalg.det, tm.linalg.inv]:
        assert np.allclose(fun(C), fun(V).x)

    for fun in [
        tm.special.dev,
        tm.special.tresca,
        tm.special.von_mises,
        tm.special.sym,
    ]:
        fun(T)

    assert tm.linalg.eigvalsh(T).shape == (3,)

    assert tm.linalg.expm(T).shape == (3, 3)

    assert tm.base.cross(F, F).shape == F.shape
    assert tm.base.eye(F).shape == F.shape
    assert np.allclose(tm.base.eye(F), tm.base.eye(T))
    assert np.allclose(tm.array(T).x, tm.array(F))
    assert np.allclose(tm.array([T, T]).x, tm.array([F, F]))
    assert np.allclose(tm.vstack([T, T]).x, tm.vstack([F, F]))
    assert np.allclose(tm.hstack([T, T]).x, tm.hstack([F, F]))
    assert np.allclose(tm.stack([T, T]).x, tm.stack([F, F]))
    assert np.allclose(tm.repeat(T, 3).x, tm.repeat(F, 3))
    assert np.allclose(tm.tile(T, 3).x, tm.tile(F, 3))
    assert np.allclose(tm.split(T, [1, 2])[1].x, tm.split(F, [1, 2])[1])
    assert np.allclose(T.squeeze().x, tm.squeeze(F))
    assert np.allclose(T[:1].squeeze().x, tm.squeeze(F[:1]))


def test_einsum():
    F = np.eye(3) + np.arange(1, 10).reshape(3, 3) / 10
    T = tr.Tensor(F)

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


def test_slice():
    F = np.eye(3) + np.arange(1, 10).reshape(3, 3) / 10
    T = tr.Tensor(F)

    T.ravel()
    T[0] = F[0]
    T[:, 0] = F[:, 0]
    T[:, 0] = T[:, 0]


def test_reshape():
    x = np.ones((3, 3, 100))
    t = tr.Tensor(x, x, x, x, ntrax=1)
    u = tr.Tensor(x, ntrax=1)

    u[0] = t[0]

    t.reshape(9)
    t.reshape(3, 3)

    tm.reshape(t, (9,))
    tm.reshape(t, (3, 3))

    tm.reshape(x, (3, 3, 100))

    tm.broadcast_to(x, x.shape)
    tm.broadcast_to(t, (*t.shape, *t.trax))


def test_eigh():
    F = np.diag([1.2, 1.2, 2.0])
    T = tr.Tensor(F)

    assert tm.linalg.eigh(T)[0].shape == (3,)
    assert tm.linalg.eigh(T)[1].shape == (3, 3, 3)

    assert tm.linalg.eigvalsh(T).shape == (3,)

    F = np.tile(F.reshape(3, 3, 1), 5)
    T = tr.Tensor(F, ntrax=1)

    assert tm.linalg.eigh(T)[0].shape == (3,)
    assert tm.linalg.eigh(T)[1].shape == (3, 3, 3)

    assert tm.linalg.eigvalsh(T).shape == (3,)


def test_triu():
    F = np.tile((np.eye(3) + np.arange(1, 10).reshape(3, 3) / 10).reshape(3, 3, 1), 10)
    V = tr.Tensor(F, F, F, ntrax=1)
    T = V.T() @ V

    t = tm.special.triu_1d(T)
    assert t.shape == (6,)
    assert t.size == 6

    U = tm.special.from_triu_1d(t)

    assert np.allclose(tr.f(T), tr.f(U))
    assert np.allclose(tr.δ(T), tr.δ(U))

    V = tm.special.from_triu_2d(np.ones((6, 6, 10)))

    assert V.shape == (3, 3, 3, 3, 10)


def test_logical():
    F = np.tile((np.eye(3) + np.arange(1, 10).reshape(3, 3) / 10).reshape(3, 3, 1), 10)
    T = tr.Tensor(F, ntrax=1)

    G = np.tile((np.eye(3) - np.arange(1, 10).reshape(3, 3) / 10).reshape(3, 3, 1), 10)
    V = tr.Tensor(G, ntrax=1)

    for A in [F, T]:
        for B in [G, V]:
            A > B
            A < B
            A >= B
            A <= B
            A == B

            assert (A > B).dtype == bool
            assert np.all(A > B)
            assert np.all(B < A)


if __name__ == "__main__":
    test_math()
    test_einsum()
    test_slice()
    test_reshape()
    test_eigh()
    test_triu()
    test_logical()
