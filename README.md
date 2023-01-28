# tensortrax

```
 _                            
| |                          ████████╗██████╗  █████╗ ██╗  ██╗
| |_ ___ _ __  ___  ___  _ __╚══██╔══╝██╔══██╗██╔══██╗╚██╗██╔╝
| __/ _ \ '_ \/ __|/ _ \| '__|  ██║   ██████╔╝███████║ ╚███╔╝ 
| ||  __/ | | \__ \ (_) | |     ██║   ██╔══██╗██╔══██║ ██╔██╗ 
 \__\___|_| |_|___/\___/|_|     ██║   ██║  ██║██║  ██║██╔╝ ██╗
                                ╚═╝   ╚═╝  ╚═╝╚═╝  ╚═╝╚═╝  ╚═╝  
```

Math on (Hyper-Dual) Tensors with Trailing Axes.

[![PyPI version shields.io](https://img.shields.io/pypi/v/tensortrax.svg)](https://pypi.python.org/pypi/tensortrax/) [![License: GPL v3](https://img.shields.io/badge/License-GPLv3-blue.svg)](https://www.gnu.org/licenses/gpl-3.0) ![Made with love in Graz (Austria)](https://img.shields.io/badge/Made%20with%20%E2%9D%A4%EF%B8%8F%20in-Graz%20(Austria)-0c674a) ![Codestyle black](https://img.shields.io/badge/code%20style-black-black) [![DOI](https://zenodo.org/badge/570708066.svg)](https://zenodo.org/badge/latestdoi/570708066) [![codecov](https://codecov.io/github/adtzlr/tensortrax/branch/main/graph/badge.svg?token=7DTH0HKYO9)](https://codecov.io/github/adtzlr/tensortrax)

# Features
- Designed to operate on input arrays with trailing axes
- Essential vector/tensor Hyper-Dual number math, including limited support for `einsum` (restricted to max. three operands)
- Forward Mode Automatic Differentiation (AD) using Hyper-Dual Tensors, up to second order derivatives
- Create functions in terms of Hyper-Dual Tensors
- Evaluate the function, the gradient (jacobian) and the hessian of scalar-valued functions or functionals on given input arrays
- Straight-forward definition of custom functions in variational-calculus notation
- Stable gradient and hessian of eigenvalues `eigvalsh` in case of repeated equal eigenvalues
- Slicing and item assignments

# Not Features
- Not imitating a full-featured NumPy (e.g. like [Autograd](https://github.com/HIPS/autograd))
- No arbitrary-order gradients (only first- and second order gradients)
- No support for `dtype=complex`

# Motivation
Compared to other Python libaries which introduce a new (hyper-) dual `dtype` (treated as `dtype=object` in NumPy), `tensortrax` relies on its own `Tensor` class. This approach involves a re-definition of all essential math operations (and NumPy-functions), whereas the `dtype`-approach supports most operations (even NumPy) out of the box. However, in `tensortrax` NumPy operates on default data types (e.g. `dtype=float`). This allows to support functions like `np.einsum()`. Beside the differences concerning the underlying `dtype`, `tensortrax` is formulated on (tensorial) calculus of variation. Gradient- and hessian-vector products are evaluated with very little overhead compared to analytic formulations.

# Usage
Let's define a scalar-valued function which operates on a tensor. The math module `tensortrax.math` provides some essential NumPy-like functions.

```python
import tensortrax as tr
import tensortrax.math as tm


def fun(F, mu=1):
    C = F.T() @ F
    I1 = tm.trace(C)
    J = tm.linalg.det(F)
    return mu / 2 * (J ** (-2 / 3) * I1 - 3)
```

The hessian of the scalar-valued function w.r.t. the chosen function argument (here, `wrt=0` or `wrt="F"`) is evaluated by variational calculus (Forward Mode AD implemented as Hyper-Dual Tensors). The function is called once for each component of the hessian (symmetry is taken care of). The function and the gradient are evaluated with no additional computational cost. Optionally, the function calls are executed in parallel (threaded).

```python
import numpy as np

# some random input data
np.random.seed(125161)
F = np.random.rand(3, 3, 8, 50) / 10
for a in range(3):
    F[a, a] += 1

# W = tr.function(fun, wrt=0, ntrax=2)(F)
# dWdF = tr.gradient(fun, wrt=0, ntrax=2)(F)
# d2WdF2, dWdF, W = tr.hessian(fun, wrt="F", ntrax=2, full_output=True)(F=F)
d2WdF2 = tr.hessian(fun, wrt="F", ntrax=2, parallel=False)(F=F)
```

# Performance
A [benchmark](https://github.com/adtzlr/tensortrax/blob/main/docs/benchmark/benchmark.py) for the gradient and hessian runtimes of an isotropic hyperelastic strain energy function demonstrates the performance of this package. The hessian is evaluated in about five seconds for one million input tensors (Intel Core i7-11850H, 32GB RAM).

```math
\psi(\boldsymbol{C}) = tr(\boldsymbol{C}) - \ln(\det(\boldsymbol{C}))
```

| Tensors | Gradient in s | Hessian in s |
| ------- | ------------- | ------------ |
|       2 |       0.00552 |      0.01474 |
|       8 |       0.00429 |      0.01420 |
|      32 |       0.00415 |      0.01364 |
|     128 |       0.00418 |      0.01453 |
|     512 |       0.00697 |      0.02465 |
|    2048 |       0.00831 |      0.03134 |
|    8192 |       0.01289 |      0.05174 |
|   32768 |       0.02837 |      0.11737 |
|  131072 |       0.16327 |      0.58191 |
|  524288 |       0.86078 |      2.65141 |
| 2097152 |       2.97900 |     10.95087 |

![benchmark](https://user-images.githubusercontent.com/5793153/214539409-63d9418e-9cb6-4e38-9a86-572665da30fe.svg)

# Theory
The calculus of variation deals with variations, i.e. small changes in functions and functionals. A small-change in a function is evaluated by applying small changes on the tensor components.

```math
\psi = \psi(\boldsymbol{F})
```

```math
\delta \psi = \delta \psi(\boldsymbol{F}, \delta \boldsymbol{F})
```

Let's take the trace of a tensor product as an example. The variation is evaluated as follows:

```math
\psi = tr(\boldsymbol{F}^T \boldsymbol{F}) = \boldsymbol{F} : \boldsymbol{F}
```

```math
\delta \psi = \delta \boldsymbol{F} : \boldsymbol{F} + \boldsymbol{F} : \delta \boldsymbol{F} = 2 \ \boldsymbol{F} : \delta \boldsymbol{F}
```

The $P_{ij}$ - component of the jacobian $\boldsymbol{P}$ is now numerically evaluated by setting the respective variational component $\delta F_{ij}$ of the tensor to one and all other components to zero. In total, $i \cdot j$ function calls are necessary to assemble the full jacobian. For example, the $12$ - component is evaluated as follows:

```math
\delta \boldsymbol{F}_{(12)} = \begin{bmatrix} 0 & 1 & 0 \\ 0 & 0 & 0 \\ 0 & 0 & 0 \end{bmatrix}
```

```math
\delta_{(12)} \psi = \frac{\partial \psi}{\partial F_{12}} = 2 \ \boldsymbol{F} : \delta \boldsymbol{F}_{(12)} = 2 \ \boldsymbol{F} : \begin{bmatrix} 0 & 1 & 0 \\ 0 & 0 & 0 \\ 0 & 0 & 0 \end{bmatrix}
```

The second order variation, i.e. a variation applied on another variation of a function is evaluated in the same way as a first order variation.

```math
\Delta \delta \psi = 2 \ \delta \boldsymbol{F} : \Delta \boldsymbol{F} + 2 \ \boldsymbol{F} : \Delta \delta \boldsymbol{F}
```

Once again, each component $A_{ijkl}$ of the fourth-order hessian is numerically evaluated. In total, $i \cdot j \cdot k \cdot l$ function calls are necessary to assemble the full hessian (without considering symmetry). For example, the $1223$ - component is evaluated by setting $\Delta \delta \boldsymbol{F} = \boldsymbol{0}$ and $\delta \boldsymbol{F}$ and $\Delta \boldsymbol{F}$ as follows:

```math
\delta \boldsymbol{F}_{(12)} = \begin{bmatrix} 0 & 1 & 0 \\ 0 & 0 & 0 \\ 0 & 0 & 0 \end{bmatrix}
```

```math
\Delta \boldsymbol{F}_{(23)} = \begin{bmatrix} 0 & 0 & 0 \\ 0 & 0 & 1 \\ 0 & 0 & 0 \end{bmatrix}
```

```math
\Delta \delta \boldsymbol{F} = \begin{bmatrix} 0 & 0 & 0 \\ 0 & 0 & 0 \\ 0 & 0 & 0 \end{bmatrix}
```

```math
\Delta_{(23)} \delta_{(12)} \psi = \Delta_{(12)} \delta_{(23)} \psi = \frac{\partial^2 \psi}{\partial F_{12}\ \partial F_{23}} 
```

```math
\Delta_{(23)} \delta_{(12)} \psi = 2 \ \delta \boldsymbol{F}_{(12)} : \Delta \boldsymbol{F}_{(23)} + 2 \ \boldsymbol{F} : \Delta \delta \boldsymbol{F}
```

# Numeric calculus of variation in `tensortrax`
Each Tensor has four attributes: the (real) tensor array and the (hyper-dual) variational arrays. To obtain the above mentioned $12$ - component of the gradient and the $1223$ - component of the hessian, a tensor has to be created with the appropriate small-changes of the tensor components (dual arrays).

```python
import tensortrax as tr
from tensortrax import Tensor, f, δ, Δ, Δδ
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
I1_C = trace(F.T() @ F)
```

The function as well as the gradient and hessian components are accessible as:

```python
ψ = f(I1_C)
P_12 = δ(I1_C)  # (= Δ(I1_C))
A_1223 = Δδ(I1_C)
```

To obtain full gradients and hessians of scalar-valued functions in one function call, `tensortrax` provides helpers (decorators) which handle the multiple function calls.

```python
fun = lambda F: trace(F.T() @ F)

func = tr.function(fun)(x)
grad = tr.gradient(fun)(x)
hess = tr.hessian(fun)(x)
```

For tensor-valued functions, use `jacobian()` instead of `gradient()`.

```python
fun = lambda F: F.T() @ F

jac = tr.jacobian(fun)(x)
```


Evaluate the gradient- as well as the hessian-vector(s)-product:

```python
gvp = tr.gradient_vector_product(fun)(x, δx=x)
hvp = tr.hessian_vector_product(fun)(x, δx=x)
hvsp = tr.hessian_vectors_product(fun)(x, δx=x, Δx=x)
```

# Extensions
Custom functions (extensions) are easy to implement in `tensortrax`. Beside the function expression, three additional (dual) variation expressions have to be defined. 

```python
import numpy as np
from tensortrax import Tensor, f, δ, Δ, Δδ


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
```

> **Hint**: *Feel free to [contribute](https://github.com/adtzlr/tensortrax/fork) missing math-functions to [`tensortrax/math/_math_tensor.py`](https://github.com/adtzlr/tensortrax/blob/main/tensortrax/math/_math_tensor.py)* :page_with_curl: :pencil2:.
