import tensortrax as tr
import tensortrax.math as tm
import numpy as np


def fun(F):
    C = F.T() @ F
    I1 = tm.trace(C)
    J = tm.det(F)
    return I1 * J - 3


def test_gradient_hessian():

    dudX = np.arange(9).reshape(3, 3) / 10
    F = tm._eye(dudX) + dudX

    ww = tr.function(fun, ntrax=0)(F)
    dwdf, w = tr.gradient(fun, ntrax=0)(F)
    d2WdF2, dWdF, W = tr.hessian(fun, ntrax=0)(F)


if __name__ == "__main__":
    test_gradient_hessian()
