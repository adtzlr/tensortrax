import tensortrax as tr
import tensortrax.math as tm
import numpy as np


def neo_hooke(F):
    C = F.T() @ F
    I1 = tm.trace(C)
    J = tm.linalg.det(F)
    return (J ** (-2 / 3) * I1 - 3) / 2


def ogden(F, mu=1, alpha=2):
    C = F.T() @ F
    J = tm.linalg.det(F)
    λ = tm.sqrt(tm.linalg.eigvalsh(J ** (-2 / 3) * C))
    return tm.sum(1 / alpha * (λ**alpha - 1))


def test_hvp():

    F = δF = ΔF = (np.eye(3).ravel() + np.arange(9) / 10).reshape(3, 3, 1, 1)

    for parallel in [False, True]:

        for fun in [neo_hooke, ogden]:
            δfun = tr.gradient_vector_product(fun, ntrax=2, parallel=parallel)(F, δF)
            Δδfun = tr.hessian_vector_product(fun, ntrax=2, parallel=parallel)(
                F, δF, ΔF
            )

            assert δfun.shape == (1, 1)
            assert Δδfun.shape == (1, 1)

            assert not np.any(np.isnan(δfun))
            assert not np.any(np.isnan(Δδfun))


if __name__ == "__main__":
    test_hvp()
