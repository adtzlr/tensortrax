import numpy as np

import tensortrax as tr


def simple(F):
    return F


def right_cauchy_green(F):
    return F.T() @ F


def test_jacobian():
    F = (np.eye(3).ravel() + np.arange(9) / 10).reshape(3, 3, 1, 1)

    for parallel in [False, True]:
        for fun in [simple, right_cauchy_green]:
            c = tr.function(fun, ntrax=2, parallel=parallel)(F)
            dCdF = tr.jacobian(fun, ntrax=2, parallel=parallel)(F)
            dCdF, C = tr.jacobian(fun, ntrax=2, parallel=parallel, full_output=True)(F)

            assert c.shape == (3, 3, 1, 1)
            assert C.shape == (3, 3, 1, 1)
            assert dCdF.shape == (3, 3, 3, 3, 1, 1)

            assert np.allclose(C, c)


if __name__ == "__main__":
    test_jacobian()
