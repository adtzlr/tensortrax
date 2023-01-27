# -*- coding: utf-8 -*-
"""
Created on Thu Jan 26 23:42:40 2023

@author: z0039mte
"""

import numpy as np

import tensortrax as tr
import tensortrax.math as tm


def W(F):
    return F.T()


F = np.tile(np.eye(3).reshape(3, 3, 1) + np.arange(9).reshape(3, 3, 1) / 10, 100)
dWdF = tr.jacobian(W, ntrax=1)(F)

tr.jacobian(tr.jacobian(W, ntrax=1), ntrax=1)(F)
