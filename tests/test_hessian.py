import tensortrax as tr
import tensortrax.math as tm
import numpy as np


def neo_hooke(F):
    C = F.T() @ F
    I1 = tm.trace(C)
    J = tm.det(F)
    return J ** (-2 / 3) * I1 - 3


def ogden(F, mu=1, alpha=2):
    C = F.T() @ F
    J = tm.det(F)
    λ = tm.sqrt(tm.eigvalsh(J ** (-2 / 3) * C))
    return tm.sum(1 / alpha * (λ**alpha - 1))


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


def test_repeated_eigvals():

    F = (np.eye(3).ravel()).reshape(3, 3, 1, 1)

    for fun in [ogden]:
        d2WdF2, dWdF, W = tr.hessian(fun, ntrax=2)(F)

        assert not np.any(np.isnan(W))
        assert not np.any(np.isnan(dWdF))
        assert not np.any(np.isnan(d2WdF2))


if __name__ == "__main__":
    test_function_gradient_hessian()
    test_repeated_eigvals()
