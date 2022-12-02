import tensortrax as tr
import tensortrax.math as tm
import numpy as np


def neo_hooke(F):
    C = F.T() @ F
    I1 = tm.trace(C)
    J = tm.det(F)
    return (J ** (-2 / 3) * I1 - 3) / 2


def ogden(F, mu=1, alpha=2):
    C = F.T() @ F
    J = tm.det(F)
    λ = tm.sqrt(tm.eigvalsh(J ** (-2 / 3) * C))
    return tm.sum(1 / alpha * (λ**alpha - 1))


def trig(F):
    C = F.T() @ F
    I1 = tm.trace(C)
    return tm.sin(I1) + tm.cos(I1) + tm.tan(I1) + tm.tanh(I1)


def test_function_gradient_hessian():

    F = (np.eye(3).ravel() + np.arange(9) / 10).reshape(3, 3, 1, 1)

    for fun in [neo_hooke, ogden]:
        ww = tr.function(fun, ntrax=2)(F)
        dwdf, w = tr.gradient(fun, ntrax=2)(F)
        d2WdF2, dWdF, W = tr.hessian(fun, ntrax=2)(F)

        assert W.shape == (1, 1)
        assert dWdF.shape == (3, 3, 1, 1)
        assert d2WdF2.shape == (3, 3, 3, 3, 1, 1)

        assert np.allclose(w, ww)
        assert np.allclose(w, W)
        assert np.allclose(dwdf, dWdF)


def test_trig():

    F = (np.eye(3).ravel() + np.arange(9) / 10).reshape(3, 3)

    for fun in [trig]:
        ww = tr.function(fun)(F)
        dwdf, w = tr.gradient(fun)(F)
        d2WdF2, dWdF, W = tr.hessian(fun)(F)


def test_repeated_eigvals():

    F = np.eye(3)

    ntrax = len(F.shape) - 2
    d2WdF2, dWdF, W = tr.hessian(ogden, ntrax=ntrax)(F)
    d2wdf2, dwdf, w = tr.hessian(neo_hooke, ntrax=ntrax)(F)

    assert np.allclose(w, W)
    assert np.allclose(dwdf, dWdF)
    assert np.allclose(d2wdf2, d2WdF2)

    F = np.eye(3)
    F[2, 2] = 2

    ntrax = len(F.shape) - 2
    d2WdF2, dWdF, W = tr.hessian(ogden, ntrax=ntrax)(F)
    d2wdf2, dwdf, w = tr.hessian(neo_hooke, ntrax=ntrax)(F)

    assert np.allclose(w, W)
    assert np.allclose(dwdf, dWdF)
    assert np.allclose(d2wdf2, d2WdF2)

    F = (np.eye(3).ravel()).reshape(3, 3, 1, 1)

    ntrax = len(F.shape) - 2
    d2WdF2, dWdF, W = tr.hessian(ogden, ntrax=ntrax)(F)
    d2wdf2, dwdf, w = tr.hessian(neo_hooke, ntrax=ntrax)(F)

    assert np.allclose(w, W)
    assert np.allclose(dwdf, dWdF)
    assert np.allclose(d2wdf2, d2WdF2)

    F = (np.eye(3).ravel()).reshape(3, 3, 1, 1)
    F[2, 2] = 2

    ntrax = len(F.shape) - 2
    d2WdF2, dWdF, W = tr.hessian(ogden, ntrax=ntrax)(F)
    d2wdf2, dwdf, w = tr.hessian(neo_hooke, ntrax=ntrax)(F)

    assert np.allclose(w, W)
    assert np.allclose(dwdf, dWdF)
    assert np.allclose(d2wdf2, d2WdF2)


if __name__ == "__main__":
    test_function_gradient_hessian()
    test_repeated_eigvals()
    test_trig()
