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
    return sum(1 / alpha * (λ**alpha - 1))


def test_tensor():

    np.random.seed(125161)
    dudX = np.random.rand(3, 3, 8, 50) / 10
    F = tm._eye(dudX) + dudX

    for fun in [neo_hooke, ogden]:
        d2WdF2, dWdF, W = tr.hessian(neo_hooke)(F)

        assert W.shape == (8, 50)
        assert dWdF.shape == (3, 3, 8, 50)
        assert d2WdF2.shape == (3, 3, 3, 3, 8, 50)


if __name__ == "__main__":
    test_tensor()
