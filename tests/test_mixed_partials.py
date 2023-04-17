# -*- coding: utf-8 -*-
"""
Created on Mon Feb 13 22:27:25 2023

@author: z0039mte
"""

import numpy as np

import tensortrax as tr
import tensortrax.math as tm


def test_mixed_partials_scalars():
    x = np.array([5.7])
    y = np.array([2.3])

    def fun(x, y):
        return tm.log(x) * tm.exp(y)

    r = tr.Tensor(x)
    s = tr.Tensor(y)

    # init Tensors with to be used for second partial derivatives
    r.init(hessian=True, δx=True, Δx=False)
    s.init(hessian=True, δx=False, Δx=True)

    f = fun(r, s)
    dfdxdy = tr.Δδ(f)
    Dfdxdy = 1 / x * np.exp(y)
    assert np.allclose(dfdxdy, Dfdxdy)

    # re-init Tensors to be used with first partial derivatives
    r.init(gradient=True, δx=True)
    s.init(gradient=True, δx=False)

    f = fun(r, s)
    dfdx = tr.δ(f)
    Dfdx = np.exp(y) / x

    assert np.allclose(dfdx, Dfdx)


def test_mixed_partials():
    x = (np.eye(3).ravel() + np.arange(9)).reshape(3, 3)
    y = (np.eye(3).ravel() + np.arange(10, 19)).reshape(3, 3)

    def fun(x, y):
        return tm.trace(x) * tm.linalg.det(y)

    r = tr.Tensor(x)
    s = tr.Tensor(y)

    # init Tensors with to be used for second partial derivatives
    r.init(hessian=True, δx=True, Δx=False)
    s.init(hessian=True, δx=False, Δx=True)

    f = fun(r, s)
    dfdxdy = tr.Δδ(f)
    Dfdxdy = np.einsum(
        "ij...,kl...->ijkl...", np.eye(3), np.linalg.det(y) * np.linalg.inv(y).T
    )

    assert np.allclose(dfdxdy, Dfdxdy)

    # re-init Tensors to be used with first partial derivatives
    r.init(gradient=True, δx=True)
    s.init(gradient=True, δx=False)

    f = fun(r, s)
    dfdx = tr.δ(f)
    Dfdx = tm.linalg.det(y) * np.eye(3)

    assert np.allclose(dfdx, Dfdx)


if __name__ == "__main__":
    test_mixed_partials_scalars()
    test_mixed_partials()
