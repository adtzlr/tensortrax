import numpy as np
import pytest

import tensortrax as tr
import tensortrax.math as tm


def test_math():
    F = np.eye(3) + np.arange(9).reshape(3, 3) / 10
    T = tr.Tensor(F)
    print(T)

    C = F.T @ F

    assert np.allclose(tr.f(T.T @ F), C)
    assert np.allclose(tr.f(F.T @ T), C)
    assert np.allclose(tr.f(T.T @ T), C)

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
    assert np.allclose(tm.linalg.pinv(F), tm.linalg.pinv(T).x)
    assert np.allclose(tm.linalg.pinv(F), tm.linalg.inv(T).x)

    tm.linalg._det(F[:2, :2])
    tm.linalg._det(F[:1, :1])

    tm.linalg.det(tr.Tensor(np.eye(4)))
    tm.linalg.det(T[:2, :2])
    tm.linalg.det(T[:1, :1])

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
        tm.special.erf,
    ]:
        assert np.allclose(fun(F), fun(T).x)

    C = F.T @ F
    V = tr.Tensor(C)

    for fun in [tm.linalg.det, tm.linalg.inv, tm.linalg.eigvalsh]:
        assert np.allclose(fun(C), fun(V).x)

    for fun in [tm.linalg.eigh]:
        assert np.all([np.allclose(x, y.x) for x, y in zip(fun(C), fun(V))])

    for fun in [
        tm.special.dev,
        tm.special.tresca,
        tm.special.von_mises,
        tm.special.sym,
    ]:
        fun(T)

    assert tm.linalg.eigvalsh(T).shape == (3,)

    assert tm.linalg.expm(T).shape == (3, 3)
    assert tm.linalg.sqrtm(T).shape == (3, 3)

    assert tm.base.cross(F, F).shape == F.shape
    assert tm.base.eye(F).shape == F.shape
    assert np.allclose(tm.base.eye(F), tm.base.eye(T))
    assert np.allclose(tm.array(T).x, tm.array(F))
    assert np.allclose(tm.array(F, like=T).x, tm.array(F))
    assert np.allclose(T.copy().x, T.x)
    assert np.allclose(tm.array([T, T]).x, tm.array([F, F]))
    assert np.allclose(tm.vstack([T, T]).x, tm.vstack([F, F]))
    assert np.allclose(tm.hstack([T, T]).x, tm.hstack([F, F]))
    assert np.allclose(tm.stack([T, T]).x, tm.stack([F, F]))
    assert np.allclose(tm.concatenate([T, T]).x, tm.concatenate([F, F]))
    assert np.allclose(tm.repeat(T, 3).x, tm.repeat(F, 3))
    assert np.allclose(tm.tile(T, 3).x, tm.tile(F, 3))
    assert np.allclose(tm.split(T, [1, 2])[1].x, tm.split(F, [1, 2])[1])
    assert np.allclose(T.squeeze().x, tm.squeeze(F))
    assert np.allclose(T[:1].squeeze().x, tm.squeeze(F[:1]))
    assert T.astype(int).x.dtype == int


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

    tm.einsum("ij...,kl...,mn...,pq...->ijklmnpq...", F, F, F, F)
    tm.einsum("ij...,kl...,mn...,pq...->ijklmnpq...", F, F, F, T)
    tm.einsum("ij...,kl...,mn...,pq...->ijklmnpq...", F, F, T, F)
    tm.einsum("ij...,kl...,mn...,pq...->ijklmnpq...", F, T, F, F)
    tm.einsum("ij...,kl...,mn...,pq...->ijklmnpq...", T, F, F, F)
    tm.einsum("ij...,kl...,mn...,pq...->ijklmnpq...", F, F, T, T)
    tm.einsum("ij...,kl...,mn...,pq...->ijklmnpq...", F, T, F, T)
    tm.einsum("ij...,kl...,mn...,pq...->ijklmnpq...", T, F, F, T)
    tm.einsum("ij...,kl...,mn...,pq...->ijklmnpq...", T, F, T, F)
    tm.einsum("ij...,kl...,mn...,pq...->ijklmnpq...", T, T, F, F)
    tm.einsum("ij...,kl...,mn...,pq...->ijklmnpq...", F, T, T, F)
    tm.einsum("ij...,kl...,mn...,pq...->ijklmnpq...", F, T, T, T)
    tm.einsum("ij...,kl...,mn...,pq...->ijklmnpq...", T, F, T, T)
    tm.einsum("ij...,kl...,mn...,pq...->ijklmnpq...", T, T, F, T)
    tm.einsum("ij...,kl...,mn...,pq...->ijklmnpq...", T, T, T, F)
    tm.einsum("ij...,kl...,mn...,pq...->ijklmnpq...", T, T, T, T)

    with pytest.raises(NotImplementedError):
        tm.einsum("ij...,kl...,mn...,pq...,rs...->ijklmnpqrs...", T, T, T, T, T)


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
    T = V.T @ V

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


def test_condition():
    F = np.tile((np.eye(3) + np.arange(-2, 7).reshape(3, 3) / 10).reshape(3, 3, 1), 10)
    T = tr.Tensor(F, ntrax=1)

    G = np.tile((np.eye(3) - np.arange(-7, 2).reshape(3, 3) / 10).reshape(3, 3, 1), 10)
    V = tr.Tensor(G, ntrax=1)

    Y = tm.if_else(F >= G, 2 * F, G / 2)
    Z = tm.if_else(T >= V, 2 * T, V / 2)

    max_array = tm.maximum(F, G)
    max_tensor = tm.maximum(T, V)

    assert np.allclose(max_array, max_tensor.x)

    min_array = tm.minimum(F, G)
    min_tensor = tm.minimum(T, V)

    assert np.allclose(min_array, min_tensor.x)

    np.allclose(Y, Z.x)

    with pytest.raises(NotImplementedError):
        tm.if_else(F >= T, 2 * F, V / 2)

    with pytest.raises(NotImplementedError):
        tm.if_else(T >= G, 2 * T, G / 2)


def test_try_stack():
    x = np.tile((np.eye(3) + np.arange(-2, 7).reshape(3, 3) / 10).reshape(3, 3, 1), 10)
    F = tr.Tensor(x, ntrax=1)

    C = F.T @ F
    C6 = tm.special.triu_1d(C)
    fallback = "my fallback"

    stacked = tm.special.try_stack([C6, C6], fallback=fallback)
    assert stacked.shape[0] == 12

    assert tm.special.try_stack([C, C6], fallback=fallback) == fallback
    with pytest.raises(ValueError):
        tm.stack([C, C6])


def test_inputs_not_modified():
    F = np.eye(3)[..., None] + 0.2 * np.random.rand(3, 3, 4)
    C = np.einsum("ki...,kj...->ij...", F, F)
    C0 = C.copy()

    def fun(C):
        λ = tm.linalg.eigvalsh(C)
        λ, M = tm.linalg.eigh(C)
        return tm.sum(λ) * tm.linalg.det(C)

    for evaluate in [tr.function, tr.gradient, tr.hessian, tr.jacobian]:
        evaluate(fun, ntrax=1)(C)
        assert np.array_equal(C, C0)

    def fun2(C):  # eigvalsh must not change C for later operations
        a = tm.linalg.det(C)
        tm.linalg.eigvalsh(C)
        return tm.linalg.det(C) - a

    assert np.all(tr.function(fun2, ntrax=1)(C) == 0)


def test_setitem():
    I = np.eye(3)

    def advanced_index(C):  # overwritten entries are constants
        D = C * 1.0
        D[[0, 1], [0, 1]] = 0.0
        return D[0, 0] + D[1, 1]

    def aliased(C):  # writing into D must not change C
        D = C + 1.0
        D[0, 0] = C[1, 1] * 2
        return tm.trace(C)

    def componentwise(C):  # W = C11**2 + 6 C00
        D = C * 0.0
        D[0, 0] = C[1, 1] ** 2
        D[[1, 2], [1, 2]] = C[0, 0] * 3
        return tm.trace(D)

    assert np.allclose(tr.gradient(advanced_index)(I), 0)
    assert np.allclose(tr.gradient(aliased)(I), I)
    assert np.allclose(tr.gradient(componentwise)(I), np.diag([6.0, 2.0, 0.0]))
    assert tr.hessian(componentwise)(I)[1, 1, 1, 1] == 2.0


def test_sum_axis():
    T = tr.Tensor(np.ones((3, 2, 4)), ntrax=1)

    assert tm.sum(T).shape == (2,)
    assert tm.sum(T, axis=1).shape == (3,)
    assert tm.sum(T, axis=-1).shape == (3,)  # last tensor axis, not a trailing axis
    assert tm.sum(T, axis=(0, -1)).shape == ()
    assert np.allclose(tm.sum(T, axis=None).x, 6)  # all tensor axes
    assert np.allclose(tm.sum(tm.trace(T[:2]), axis=None).x, 2)  # scalar: unchanged

    for axis in [2, -3, (0, 2)]:
        with pytest.raises(IndexError):
            tm.sum(T, axis=axis)

    with pytest.raises(IndexError):
        tm.sum(tm.trace(T[:2]))  # a scalar tensor has no axis 0

    # gradient of sum_i (sum_j C_ij)^2 = 2 (sum_k C_ik) for each column j
    C = np.random.rand(3, 3)
    for axis in [1, -1]:
        g = tr.gradient(lambda C: tm.sum(tm.sum(C, axis=axis) ** 2))(C)
        assert np.allclose(g, 2 * np.sum(C, axis=1)[:, None] * np.ones((1, 3)))

    # gradient of (sum_ij C_ij)^2 = 2 sum(C) for all entries, also with trailing axes
    X = np.random.rand(3, 3, 4)
    g = tr.gradient(lambda C: tm.sum(C, axis=None) ** 2, ntrax=1)(X)
    assert np.allclose(g, 2 * np.sum(X, axis=(0, 1)) * np.ones((3, 3, 1)))


def test_eig_nonsymmetric_variations():
    "With sym=False, eigvalsh and eigh must only depend on the symmetric part."
    np.random.seed(4)
    F = np.eye(3) + 0.2 * np.random.uniform(-1, 1, (3, 3))
    S = lambda C: (C + C.T) / 2

    for C in [F.T @ F, np.diag([2.25, 1 / 1.5, 1 / 1.5])]:  # distinct, repeated
        H = tr.hessian(lambda C: tm.sum(tm.linalg.eigvalsh(C)))(C)  # = tr(C)
        assert np.allclose(H, 0, atol=1e-6)

        H = tr.hessian(lambda C: tm.sum(tm.linalg.eigvalsh(C) ** 2))(C)  # = S:S
        Href = tr.hessian(lambda C: tm.special.ddot(S(C), S(C)))(C)
        assert np.allclose(H, Href, atol=1e-6)

    H = tr.hessian(lambda C: tm.sum(tm.linalg.eigvalsh(C) ** 1.5))(F.T @ F)
    assert np.allclose(H, np.transpose(H, (2, 3, 0, 1)))  # major symmetry

    sqrtm = tm.linalg.sqrtm
    H = tr.hessian(lambda C: tm.trace(sqrtm(C) @ sqrtm(C)))(F.T @ F)  # = tr(C)
    assert np.allclose(H, 0, atol=1e-6)


def test_repeated_eigenvalues_45deg():
    "Repeated eigenvalue, distinct eigenvector along (1, -1, 0) (uniaxial at 45°)."
    a = np.array([1.0, -1.0, 0.0]) / np.sqrt(2)
    C = (2.25 - 1 / 1.5) * np.outer(a, a) + np.eye(3) / 1.5

    ogden = lambda C: tm.sum(tm.linalg.det(C) ** (-1 / 3) * tm.linalg.eigvalsh(C))
    neo_hooke = lambda C: tm.linalg.det(C) ** (-1 / 3) * tm.trace(C)  # = ogden

    H = tr.hessian(ogden, sym=True)(C)
    assert np.allclose(H, tr.hessian(neo_hooke, sym=True)(C), atol=1e-7)


def test_eigvalsh_scale():
    "The perturbation is relative to the magnitude of the tensor."
    a = np.ones(3) / np.sqrt(3)
    B = (2.25 - 1 / 1.5) * np.outer(a, a) + np.eye(3) / 1.5

    for scale in [1e-6, 1e6]:
        C = scale * B
        λ = tr.function(tm.linalg.eigvalsh)(C)
        assert np.allclose(λ, np.linalg.eigvalsh(C), rtol=1e-7, atol=0)

        H = tr.hessian(lambda C: tm.sum(tm.linalg.eigvalsh(C) ** 2), sym=True)(C)
        Href = tr.hessian(lambda C: tm.special.ddot(C, C), sym=True)(C)
        assert np.allclose(H, Href, rtol=1e-7)


def test_constant_arrays():
    "Arrays with the shape of the tensor are constants, aligned with the tensor axes."
    np.random.seed(3)
    F = np.eye(3)[..., None] + 0.2 * np.random.uniform(-1, 1, (3, 3, 4))
    C = np.einsum("ki...,kj...->ij...", F, F)
    M = np.random.rand(3, 3)
    W = np.random.rand(3, 3, 4)  # tensor and batch axes

    eye, ddot = tm.base.eye, tm.special.ddot
    cases = [
        (
            lambda C: ddot(C - np.eye(3), C - np.eye(3)),
            lambda C: ddot(C - eye(C), C - eye(C)),
        ),
        (lambda C: ddot(np.eye(3) - C, C), lambda C: ddot(eye(C) - C, C)),
        (
            lambda C: tm.sum(tm.sum(C * M)) ** 2,
            lambda C: tm.einsum("ij...,ij->...", C, M) ** 2,
        ),
        (
            lambda C: ddot(C * W, C),
            lambda C: tm.einsum("ij...,ij...,ij...->...", C, W, C),
        ),
    ]

    for fun, ref in cases:
        for ntrax, X in [(0, C[..., 0]), (1, C)]:
            if ntrax == 0 and fun is cases[3][0]:
                continue
            assert np.allclose(
                tr.function(fun, ntrax=ntrax)(X), tr.function(ref, ntrax=ntrax)(X)
            )
            for sym in [False, True]:
                for evaluate in [tr.gradient, tr.hessian]:
                    a = evaluate(fun, ntrax=ntrax, sym=sym)(X)
                    b = evaluate(ref, ntrax=ntrax, sym=sym)(X)
                    assert np.allclose(*np.broadcast_arrays(a, b))

    mask = tr.function(lambda C: tm.if_else(C > np.eye(3), C, 2 * C), ntrax=1)(C)
    assert np.allclose(mask, np.where(C > np.eye(3)[..., None], C, 2 * C))


def test_batch_arrays():
    "Arrays with values per point (batch axes only) are unchanged."
    C = np.random.rand(3, 3, 4) + np.eye(3)[..., None]
    w = np.random.rand(4)
    fun = lambda C, w: tm.special.ddot(C * w, C)
    ref = lambda C, w: tm.einsum("ij...,...,ij...->...", C, w, C)
    for sym in [False, True]:
        assert np.allclose(
            tr.gradient(fun, ntrax=1, sym=sym)(C, w),
            tr.gradient(ref, ntrax=1, sym=sym)(C, w),
        )


def test_if_else():
    "if_else selects values and dual data, without broadcasting them to each other."
    np.random.seed(5)
    F = np.eye(3)[..., None] + 0.1 * np.random.uniform(-1, 1, (3, 3, 100))
    C = np.einsum("ki...,kj...->ij...", F, F)
    I1 = np.trace(C)
    w = np.full_like(I1, np.median(I1))
    shapes = []

    def fun(C):
        I1 = tm.trace(C)
        I1max = tm.maximum(I1, tm.array(w, like=I1))
        shapes.append((I1.x.shape, I1max.x.shape))
        return I1max**2

    assert np.allclose(tr.function(fun, ntrax=1)(C), np.maximum(I1, w) ** 2)

    dWdC = tr.gradient(fun, ntrax=1, sym=True)(C)
    assert np.allclose(dWdC, 2 * I1 * (I1 > w) * np.eye(3)[..., None])

    d2WdCdC = tr.hessian(fun, ntrax=1, sym=True)(C)
    II = np.einsum("ij,kl->ijkl", np.eye(3), np.eye(3))[..., None]
    assert np.allclose(d2WdCdC, 2 * (I1 > w) * II)

    # the values of the selection are not broadcasted to the dual data
    assert all(before == after for before, after in shapes)

    # arrays: the mask is broadcasted
    a, b = np.random.rand(3, 3, 5), np.random.rand(3, 3, 5)
    assert np.allclose(tm.if_else(a > b, a, b), np.maximum(a, b))
    assert np.allclose(
        tm.if_else((a > b)[..., :1], a, b), np.where((a > b)[..., :1], a, b)
    )


if __name__ == "__main__":
    test_math()
    test_einsum()
    test_slice()
    test_reshape()
    test_eigh()
    test_triu()
    test_logical()
    test_condition()
    test_try_stack()
    test_inputs_not_modified()
    test_setitem()
    test_sum_axis()
    test_eig_nonsymmetric_variations()
    test_repeated_eigenvalues_45deg()
    test_eigvalsh_scale()
    test_constant_arrays()
    test_batch_arrays()
    test_if_else()
