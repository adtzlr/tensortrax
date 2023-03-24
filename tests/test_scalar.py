import numpy as np
import pytest

import tensortrax as tr
import tensortrax.math as tm


def fun(x, y):
    return x**2 / y + x * tm.log(y)


def test_scalar():
    np.random.seed(6574)
    x = np.random.rand(100)

    np.random.seed(54234)
    y = np.random.rand(100)

    with pytest.raises(TypeError):
        tr.hessian(fun, wrt=[1, 2])(x, y)

    h, g, f = tr.hessian(fun, wrt=0, ntrax=1, full_output=True)(x, y)

    assert np.allclose(g, 2 * x / y + np.log(y))
    assert np.allclose(h, 2 / y)

    h, g, f = tr.hessian(fun, wrt="y", ntrax=1, full_output=True)(x=x, y=y)

    assert np.allclose(g, -(x**2) / y**2 + x / y)
    assert np.allclose(h, 2 * x**2 / y**3 - x / y**2)


if __name__ == "__main__":
    test_scalar()
