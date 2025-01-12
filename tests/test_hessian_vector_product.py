import numpy as np

import tensortrax as tr
import tensortrax.math as tm


def simple(F):
    return tm.trace(F)


def neo_hooke(F):
    C = F.T @ F
    I1 = tm.trace(C)
    J = tm.linalg.det(F)
    return (J ** (-2 / 3) * I1 - 3) / 2


def ogden(F, mu=1, alpha=2):
    C = F.T @ F
    J = tm.linalg.det(F)
    λ = tm.sqrt(tm.linalg.eigvalsh(J ** (-2 / 3) * C))
    return tm.sum(1 / alpha * (λ**alpha - 1))


def test_hvp():
    F = δF = ΔF = np.tile(
        (np.eye(3).ravel() + np.arange(9) / 10).reshape(3, 3, 1, 1), (1, 1, 2, 4)
    )

    for parallel in [False, True]:
        for fun in [simple, neo_hooke, ogden]:
            δfun = tr.gradient_vector_product(fun, wrt="F", ntrax=2, parallel=parallel)(
                F=F, δx=δF
            )
            Δδfun = tr.hessian_vectors_product(
                fun, wrt="F", ntrax=2, parallel=parallel
            )(F=F, δx=δF, Δx=ΔF)

            hvp = tr.hessian_vector_product(fun, wrt="F", ntrax=2, parallel=parallel)(
                F=F, δx=δF
            )

            assert δfun.shape == (2, 4)
            assert Δδfun.shape == (2, 4)

            assert hvp.shape == (3, 3, 2, 4)

            assert not np.any(np.isnan(δfun))
            assert not np.any(np.isnan(Δδfun))
            assert not np.any(np.isnan(hvp))


def test_gvp():
    F = np.diag([2, 1.15, 1.15])
    F[:, 0] += np.array([0, -0.15, -0.15])
    δF = np.vstack([np.zeros(3), np.zeros(3), np.array([-0.04, 0.04, 0.04])])

    δψ = tr.gradient_vector_product(neo_hooke)(F, δx=δF)
    δψ_reference = tm.special.ddot(tr.gradient(neo_hooke)(F), δF)

    assert np.isclose(δψ, δψ_reference)


if __name__ == "__main__":
    test_hvp()
    test_gvp()
