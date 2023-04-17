import numpy as np

import tensortrax as tr
import tensortrax.math as tm


def neo_hooke(F):
    C = F.T() @ F
    I1 = tm.trace(C)
    J = tm.linalg.det(F)
    return (J ** (-2 / 3) * I1 - 3) / 2


def neo_hooke_sym(C):
    C = (C + C.T()) / 2
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
    C = F.T() @ F
    J = tm.linalg.det(F)
    λ = tm.sqrt(tm.linalg.eigvalsh(J ** (-2 / 3) * C))
    return tm.sum(1 / alpha * (λ**alpha - 1))


def trig(F):
    C = F.T() @ F
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


if __name__ == "__main__":
    test_function_gradient_hessian()
    test_repeated_eigvals()
    test_trig()
    test_sym()
