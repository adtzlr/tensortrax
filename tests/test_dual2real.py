import numpy as np

import tensortrax as tr
import tensortrax.math as tm


def test_dual_to_real():
    np.random.seed(34563)
    x = (np.random.rand(3, 3) - 0.5) / 10 + np.eye(3)

    # init a Tensor with `hessian=True`
    F = tr.Tensor(x)
    F.init(hessian=True)

    # perform some math operations
    C = F.T @ F
    J = tm.linalg.det(F)
    W = tm.trace(J ** (-2 / 3) * C) - 3
    eta = 1 - 1 / 3 * tm.tanh(W / 8)

    # set old dual data as new real values (i.e. obtain the gradient)
    P = W.dual2real(like=F)
    p = tm.dual2real(W, like=F)

    P = W.dual_to_real(like=F)
    p = tm.dual_to_real(W, like=F)

    assert np.allclose(tr.f(P), tr.f(p))
    assert np.allclose(tr.δ(P), tr.δ(p))

    # perform some more math with a derived Tensor involved
    Q = eta * P

    # take the gradient
    A = tr.δ(Q)

    assert P.shape == (3, 3)
    assert Q.shape == (3, 3)
    assert A.shape == (3, 3, 3, 3)

    # can't take more than two gradients
    assert np.any(tr.Δδ(Q))


def test_real_to_dual():
    np.random.seed(34563)
    x = (np.random.rand(3, 3) - 0.5) / 10 + np.eye(3)

    # init a Tensor with `hessian=True`
    F = tr.Tensor(x)
    F.init(gradient=True)

    # perform some math operations
    C = F.T @ F
    J = tm.linalg.det(F)
    W = tm.trace(J ** (-2 / 3) * C) - 3
    η = 1 - 1 / 3 * tm.tanh(W / 8)

    # set the variation to δη * W (and the linearization to Δη * δW + η * ΔδW)
    P = tr.δ(η.real_to_dual(W))
    assert P.shape == (3, 3)


if __name__ == "__main__":
    test_dual_to_real()
    test_real_to_dual()
