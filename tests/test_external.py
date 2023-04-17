# -*- coding: utf-8 -*-
"""
Created on Mon Feb 13 22:27:25 2023

@author: z0039mte
"""

import numpy as np

import tensortrax as tr
import tensortrax.math as tm


def W(F, mu=1):
    C = F.T() @ F
    I1 = tm.trace(C)
    return mu * (I1 - 3) / 2


def neo_hooke_ext(F):
    J = tm.linalg.det(F)
    return tm.external(
        x=J ** (-1 / 3) * F,
        function=tr.function(W, ntrax=F.ntrax),
        gradient=tr.gradient(W, ntrax=F.ntrax),
        hessian=tr.hessian(W, ntrax=F.ntrax),
    )


def neo_hooke(F, mu=1):
    C = F.T() @ F
    I1 = tm.trace(C)
    J = tm.linalg.det(F)
    return mu * (J ** (-2 / 3) * I1 - 3) / 2


def test_external():
    F = np.tile((np.eye(3).ravel() + np.arange(9) / 10).reshape(3, 3, 1, 1), 2100)

    W = []
    dWdF = []
    d2WdF2 = []

    for fun in [neo_hooke, neo_hooke_ext]:
        W.append(tr.function(fun, ntrax=2)(F))
        dWdF.append(tr.gradient(fun, ntrax=2)(F))
        d2WdF2.append(tr.hessian(fun, wrt="F", ntrax=2)(F=F))

        assert W[-1].shape == (1, 2100)
        assert dWdF[-1].shape == (3, 3, 1, 2100)
        assert d2WdF2[-1].shape == (3, 3, 3, 3, 1, 2100)

    assert np.allclose(*W)
    assert np.allclose(*dWdF)
    assert np.allclose(*d2WdF2)


if __name__ == "__main__":
    test_external()
