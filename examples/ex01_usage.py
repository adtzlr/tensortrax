r"""
Quickstart
----------
Let's define a scalar-valued function which operates on a tensor. The math module
:mod:`tensortrax.math` provides some essential NumPy-like functions including linear
algebra.
"""
import tensortrax as tr
import tensortrax.math as tm


def fun(F, mu=1):
    C = F.T @ F
    I1 = tm.trace(C)
    J = tm.linalg.det(F)
    return mu / 2 * (J ** (-2 / 3) * I1 - 3)


# %%
# The Hessian of the scalar-valued function w.r.t. the chosen function argument (here,
# ``wrt=0`` or ``wrt="F"``) is evaluated by variational calculus (Forward Mode AD
# implemented as Hyper-Dual Tensors). The function is called once for each component of
# the hessian (symmetry is taken care of). The function and the gradient are evaluated
# with no additional computational cost. Optionally, the function, gradient and Hessian
# calls are executed in parallel (threaded).
import numpy as np

# some random input data
np.random.seed(125161)
F = (np.eye(3) + np.random.rand(50, 8, 3, 3) / 10).T

# different ways on how to evaluate the function
W = tr.function(fun, wrt=0, ntrax=2)(F)
dWdF = tr.gradient(fun, wrt=0, ntrax=2)(F)
d2WdF2, dWdF, W = tr.hessian(fun, wrt="F", ntrax=2, full_output=True)(F=F)
d2WdF2 = tr.hessian(fun, wrt="F", ntrax=2, parallel=False)(F=F)

# %%
# Another possibility is to define and operate on Tensors manually. This enables more
# flexible coding, which wouldn't be possible with the builtin functions. The Hu-Washizu
# Three-Field-Variational principle for nearly incompressible hyperelastic solids [1]_
# is used here to obtain mixed partial derivatives. Some random input arrays are
# generated and a Tensor is created for each variable. After performing some math, the
# hessian of the resulting tensor object is extracted.

# some random input data
n = 10
x = (np.eye(3) + np.random.rand(n, 3, 3) / 10).T
y = np.random.rand(n)
z = np.random.rand(n) / 10 + 1

# create tensors
F = tr.Tensor(x, ntrax=1)
p = tr.Tensor(y, ntrax=1)
J = tr.Tensor(z, ntrax=1)


def neo_hooke(F, mu=1):
    "Strain energy function of the Neo-Hookean material formulation."
    C = F.T @ F
    I3 = tm.linalg.det(C)
    return mu * (I3 ** (-1 / 3) * tm.trace(C) - 3) / 2


def volumetric(J, bulk=20):
    "Volumetric strain energy function."
    return bulk * (J - 1) ** 2 / 2


def W(F, p, J):
    "Hu-Washizu (u, p, J) - Three-Field-Variation."
    detF = tm.linalg.det(F)
    return neo_hooke(F) + volumetric(J) + p * (detF - J)


# init Tensors to be used with second partial derivatives
F.init(hessian=True, δx=False, Δx=False)
p.init(hessian=True, δx=True, Δx=False)
J.init(hessian=True, δx=False, Δx=True)

# evaluate a mixed second partial derivative
dWdpdJ = tr.Δδ(W(F, p, J))

# %%
# In a similar way, the gradient may be obtained by initiating a Tensor with the
# gradient argument.

# init Tensors to be used with first partial derivatives
F.init(gradient=True, δx=False)
p.init(gradient=True, δx=True)
J.init(gradient=True, δx=False)

# evaluate a partial derivative
dWdp = tr.δ(W(F, p, J))

# %%
# References
# ~~~~~~~~~~
# .. [1] J. Bonet and R. D. Wood, Nonlinear Continuum Mechanics for Finite Element
#    Analysis, 2nd ed. Cambridge: Cambridge University Press, 2008, doi:
#    `10.1017/CBO9780511755446 <https://doi.org/10.1017/CBO9780511755446>`_.
