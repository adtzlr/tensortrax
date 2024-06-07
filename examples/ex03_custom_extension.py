r"""
Custom Extensions
-----------------
Custom functions (extensions) are easy to implement in `tensortrax`. Beside the function
expression, three additional (dual) variation expressions have to be defined. 
"""
import numpy as np

from tensortrax import Tensor, Δ, Δδ, f, δ


def sin(A):
    return Tensor(
        x=np.sin(f(A)),
        δx=np.cos(f(A)) * δ(A),
        Δx=np.cos(f(A)) * Δ(A),
        Δδx=-np.sin(f(A)) * δ(A) * Δ(A) + np.cos(f(A)) * Δδ(A),
        ntrax=A.ntrax,
    )


x = np.eye(3)
y = sin(Tensor(x))

# %%
# .. note::
#    Contrary to NumPy's ``w, v = np.linalg.eigh(C)``, which returns eigenvalues and
#    -vectors, the differentiable ``w, M = tm.linalg.eigh(C)`` function returns
#    eigenvalues and eigenbases of symmetric real-valued tensors.
#
# .. tip::
#    Feel free to `contribute <https://github.com/adtzlr/tensortrax/fork>`_ missing
#    math-functions to `src/tensortrax/math/_math_tensor.py <https://github.com/adtzlr/tensortrax/blob/main/src/tensortrax/math/_math_tensor.py>`_ 📃 ✏️.
