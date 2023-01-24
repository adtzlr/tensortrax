import numpy as np

import tensortrax as tr
import tensortrax.math as tm


def neo_hooke(F):
    C = F.T() @ F
    I1 = tm.trace(C)
    J = tm.linalg.det(F)
    return (J ** (-2 / 3) * I1 - 3) / 2, F


def test_take():
    F = (np.eye(3).ravel() + np.arange(9) / 10).reshape(3, 3)

    P = tr.gradient(tr.take(neo_hooke, item=0))(F)
    A = tr.hessian(tr.take(neo_hooke, item=0))(F)

    assert P.shape == (3, 3)
    assert A.shape == (3, 3, 3, 3)

    G = tr.function(tr.take(neo_hooke, item=1))(F)
    dFdF = tr.jacobian(tr.take(neo_hooke, item=1))(F)

    assert np.allclose(F, G)
    assert np.allclose(dFdF, np.einsum("ik,jl", np.eye(3), np.eye(3)))


if __name__ == "__main__":
    test_take()
