# -*- coding: utf-8 -*-
"""
Created on Mon Feb 13 22:27:25 2023

@author: z0039mte
"""

import numpy as np

import tensortrax as tr
import tensortrax.math as tm


def test_mixed_partials():
    x = (np.eye(3).ravel() + np.arange(9)).reshape(3, 3)
    y = (np.eye(3).ravel() + np.arange(10, 19)).reshape(3, 3)

    fun = lambda x, y: tm.trace(x) * tm.linalg.det(y)

    r = tr.Tensor(x)
    r.init(hessian=True, Δx=False)

    s = tr.Tensor(y)
    s.init(hessian=True, δx=False)

    f = fun(r, s)
    dfdxdy = tr.Δδ(f)

    Dfdxdy = np.einsum(
        "ij...,kl...->ijkl...", np.eye(3), np.linalg.det(y) * np.linalg.inv(y).T
    )

    assert np.allclose(dfdxdy, Dfdxdy)


if __name__ == "__main__":
    test_mixed_partials()
