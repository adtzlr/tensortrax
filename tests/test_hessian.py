import numpy as np
import pytest

import tensortrax as tr
import tensortrax.math as tm


def neo_hooke(F):
    C = F.T @ F
    I1 = tm.trace(C)
    J = tm.linalg.det(F)
    return (J ** (-2 / 3) * I1 - 3) / 2


def neo_hooke_sym(C):
    C = (C + C.T) / 2
    I3 = tm.linalg.det(C)
    I1 = tm.trace(C)
    return (I3 ** (-1 / 3) * I1 - 3) / 2


def neo_hooke_sym_triu(C, statevars):
    tm.special.from_triu_1d(tm.special.triu_1d(C), like=C)
    sv = tm.special.from_triu_1d(statevars, like=C)
    I3 = tm.linalg.det(C)
    I1 = tm.trace(C)
    C @ sv
    return (I3 ** (-1 / 3) * I1 - 3) / 2


def ogden(F, mu=1, alpha=2):
    C = F.T @ F
    J = tm.linalg.det(F)
    λ = tm.sqrt(tm.linalg.eigvalsh(J ** (-2 / 3) * C))
    return tm.sum(1 / alpha * (λ**alpha - 1))


def trig(F):
    C = F.T @ F
    I1 = tm.trace(C)
    return tm.sin(I1) + tm.cos(I1) + tm.tan(I1) + tm.tanh(I1)


def test_function_gradient_hessian():
    F = np.tile((np.eye(3).ravel() + np.arange(9) / 10).reshape(3, 3, 1, 1), 2100)

    for parallel in [False, True]:
        for fun in [neo_hooke, ogden]:
            ww = tr.function(fun, ntrax=2, parallel=parallel)(F)
            dwdf, w = tr.gradient(fun, ntrax=2, parallel=parallel, full_output=True)(F)
            d2WdF2, dWdF, W = tr.hessian(
                fun, wrt="F", ntrax=2, parallel=parallel, full_output=True
            )(F=F)
            assert W.shape == (1, 2100)
            assert dWdF.shape == (3, 3, 1, 2100)
            assert d2WdF2.shape == (3, 3, 3, 3, 1, 2100)

            assert np.allclose(w, ww)
            assert np.allclose(w, W)
            assert np.allclose(dwdf, dWdF)


def test_trig():
    F = (np.eye(3).ravel() + np.arange(9) / 10).reshape(3, 3)

    for fun in [trig]:
        ww = tr.function(fun)(F)
        dwdf = tr.gradient(fun, full_output=False)(F)
        d2WdF2 = tr.hessian(fun, full_output=False)(F)

        assert ww.shape == ()
        assert dwdf.shape == (3, 3)
        assert d2WdF2.shape == (3, 3, 3, 3)


def test_repeated_eigvals():
    F = np.eye(3)

    ntrax = len(F.shape) - 2
    d2WdF2, dWdF, W = tr.hessian(ogden, ntrax=ntrax, full_output=True)(F)
    d2wdf2, dwdf, w = tr.hessian(neo_hooke, ntrax=ntrax, full_output=True)(F)

    assert np.allclose(w, W)
    assert np.allclose(dwdf, dWdF)
    assert np.allclose(d2wdf2, d2WdF2)

    F = np.eye(3)
    F[2, 2] = 2

    ntrax = len(F.shape) - 2
    d2WdF2, dWdF, W = tr.hessian(ogden, ntrax=ntrax, full_output=True)(F)
    d2wdf2, dwdf, w = tr.hessian(neo_hooke, ntrax=ntrax, full_output=True)(F)

    assert np.allclose(w, W)
    assert np.allclose(dwdf, dWdF)
    assert np.allclose(d2wdf2, d2WdF2)

    F = (np.eye(3).ravel()).reshape(3, 3, 1, 1)

    ntrax = len(F.shape) - 2
    d2WdF2, dWdF, W = tr.hessian(ogden, ntrax=ntrax, full_output=True)(F)
    d2wdf2, dwdf, w = tr.hessian(neo_hooke, ntrax=ntrax, full_output=True)(F)

    assert np.allclose(w, W)
    assert np.allclose(dwdf, dWdF)
    assert np.allclose(d2wdf2, d2WdF2)

    F = (np.eye(3).ravel()).reshape(3, 3, 1, 1)
    F[2, 2] = 2

    ntrax = len(F.shape) - 2
    d2WdF2, dWdF, W = tr.hessian(ogden, ntrax=ntrax, full_output=True)(F)
    d2wdf2, dwdf, w = tr.hessian(neo_hooke, ntrax=ntrax, full_output=True)(F)

    assert np.allclose(w, W)
    assert np.allclose(dwdf, dWdF)
    assert np.allclose(d2wdf2, d2WdF2)


def test_sym():
    F = (np.eye(3).ravel() + np.arange(9) / 10).reshape(3, 3)
    C = F.T @ F
    statevars = tm.special.triu_1d(C)

    S = 2 * tr.gradient(neo_hooke_sym)(C)
    D = 4 * tr.hessian(neo_hooke_sym)(C)

    s = 2 * tr.gradient(neo_hooke_sym_triu, wrt="C", sym=True)(C=C, statevars=statevars)
    d = 4 * tr.hessian(neo_hooke_sym_triu, sym=True)(C, statevars=statevars)

    assert np.allclose(S, s)
    assert np.allclose(D, d)


def test_parallel():
    "parallel=True must give the same results as parallel=False (needs >= 2 cores)."
    np.random.seed(0)
    q, c = 4, 500
    F = np.eye(3)[..., None, None] + 0.1 * np.random.uniform(-1, 1, (3, 3, q, c))
    C = np.einsum("ki...,kj...->ij...", F, F)

    def neo_hooke(C, mu):
        return mu * (tm.linalg.det(C) ** (-1 / 3) * tm.trace(C) - 3)

    for mu in [1.5, np.random.rand(q, c), np.random.rand(1, c), np.random.rand(c)]:
        for evaluate in [tr.function, tr.gradient, tr.hessian]:
            serial = evaluate(neo_hooke, ntrax=2)(C, mu=mu)
            parallel = evaluate(neo_hooke, ntrax=2, parallel=True)(C, mu=mu)
            assert serial.shape == parallel.shape
            assert np.allclose(serial, parallel)

    # constant derivatives are compressed along the batch axes
    for evaluate, fun in [(tr.hessian, tm.trace), (tr.jacobian, lambda C: 2 * C)]:
        serial = evaluate(fun, ntrax=2)(C)
        parallel = evaluate(fun, ntrax=2, parallel=True)(C)
        assert serial.shape == parallel.shape
        assert np.allclose(serial, parallel)


def test_structural_zeros():
    "Dual data which is not required is not tracked (structural zeros)."
    C = np.random.rand(3, 3, 4) + 2 * np.eye(3)[..., None]
    duals = []

    def fun(C):
        duals.append(tuple(type(a).__name__ for a in (C.δx, C.Δx, C.Δδx)))
        return tm.trace(C)

    tr.function(fun, ntrax=1)(C)
    tr.gradient(fun, ntrax=1)(C)
    tr.hessian(fun, ntrax=1)(C)

    assert duals[0] == ("Zero", "Zero", "Zero")
    assert duals[1] == ("ndarray", "Zero", "Zero")
    assert duals[2] == ("ndarray", "ndarray", "Zero")

    zero = tr._tensor.Zero()
    repr(zero)

    with pytest.raises(TypeError):
        np.array(zero)

    zero - 3
    3 - zero
    C * zero
    zero * C

    tm.diagonal(zero)
    tm.diagonal(C * zero)

    new_zero = tm.einsum("ij...,ij...,ij...,...->...", C, C, C, zero)
    assert isinstance(new_zero, type(zero))


def test_structural_zeros_mixed():
    "Structural zeros combined with tensors with arrays of zeros as dual data."
    np.random.seed(1)
    C = np.random.rand(3, 3, 4) + 2 * np.eye(3)[..., None]
    w = np.random.rand(4) + 10  # larger than tr(C)

    def stack(C):
        return tm.sum(tm.stack([C[0, 0], C[1, 1]]) ** 2)

    def stack_ref(C):
        return C[0, 0] ** 2 + C[1, 1] ** 2

    def setitem(C):
        D = C * 1.0
        D[0, 0] = tm.array(w, like=tm.trace(C))
        return tm.special.ddot(D, C)

    def maximum(C):
        return tm.maximum(tm.trace(C), tm.array(w, like=tm.trace(C))) * tm.trace(C)

    I = np.eye(3)[..., None]
    E00 = np.zeros((3, 3, 1))
    E00[0, 0] = 1

    for evaluate in [tr.function, tr.gradient, tr.hessian]:
        assert np.allclose(evaluate(stack, ntrax=1)(C), evaluate(stack_ref, ntrax=1)(C))

    # setitem: 2 C, except the entry 00 (w)
    assert np.allclose(tr.gradient(setitem, ntrax=1)(C), (2 * C) * (1 - E00) + w * E00)

    # maximum: w I (w > tr(C))
    assert np.allclose(tr.gradient(maximum, ntrax=1)(C), w * I)
    assert np.allclose(tr.function(maximum, ntrax=1)(C), w * np.trace(C))


if __name__ == "__main__":
    test_function_gradient_hessian()
    test_repeated_eigvals()
    test_trig()
    test_sym()
    test_parallel()
    test_structural_zeros()
    test_structural_zeros_mixed()
