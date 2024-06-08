r"""
Numeric calculus of variation
-----------------------------
Each Tensor has four attributes: the (real) tensor array and the (hyper-dual)
variational arrays. To obtain the :math:`12` - component of the gradient and the
:math:`1223` - component of the hessian, a tensor has to be created with the appropriate
small-changes of the tensor components (dual arrays).
"""
import numpy as np

import tensortrax as tr
from tensortrax import Tensor, Δ, Δδ, f, δ
from tensortrax.math import trace

δF_12 = np.array(
    [
        [0, 1, 0],
        [0, 0, 0],
        [0, 0, 0],
    ],
    dtype=float,
)

ΔF_23 = np.array(
    [
        [0, 0, 0],
        [0, 0, 1],
        [0, 0, 0],
    ],
    dtype=float,
)

x = np.eye(3) + np.arange(9).reshape(3, 3) / 10
F = Tensor(x=x, δx=δF_12, Δx=ΔF_23, Δδx=None)
I1_C = trace(F.T @ F)

# %%
# The function as well as the gradient and hessian components are accessible with
# helpers.
ψ = f(I1_C)
P_12 = δ(I1_C)
A_1223 = Δδ(I1_C)

# %%
# To obtain full gradients and hessians of scalar-valued functions in one function call,
# ``tensortrax`` provides helpers (decorators) which handle the multiple function calls.
fun = lambda F: trace(F.T @ F)

func = tr.function(fun)(x)
grad = tr.gradient(fun)(x)
hess = tr.hessian(fun)(x)

# %%
# For tensor-valued functions, use ``jacobian()`` instead of ``gradient()``.
fun = lambda F: F.T @ F
jac = tr.jacobian(fun)(x)

# %%
# Evaluate the gradient- as well as the hessian-vector(s)-product.
gvp = tr.gradient_vector_product(fun)(x, δx=x)
hvp = tr.hessian_vector_product(fun)(x, δx=x)
hvsp = tr.hessian_vectors_product(fun)(x, δx=x, Δx=x)
