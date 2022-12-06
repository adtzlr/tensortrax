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

[![PyPI version shields.io](https://img.shields.io/pypi/v/tensortrax.svg)](https://pypi.python.org/pypi/tensortrax/) [![License: GPL v3](https://img.shields.io/badge/License-GPLv3-blue.svg)](https://www.gnu.org/licenses/gpl-3.0) ![Made with love in Graz (Austria)](https://img.shields.io/badge/Made%20with%20%E2%9D%A4%EF%B8%8F%20in-Graz%20(Austria)-0c674a) [![DOI](https://zenodo.org/badge/570708066.svg)](https://zenodo.org/badge/latestdoi/570708066) [![codecov](https://codecov.io/github/adtzlr/tensortrax/branch/main/graph/badge.svg?token=7DTH0HKYO9)](https://codecov.io/github/adtzlr/tensortrax)

# Features
- Designed to operate on input arrays with trailing axes
- Essential vector/tensor Hyper-Dual number math, including limited support for `einsum` (restricted to max. two operands)
- Forward Mode Automatic Differentiation (AD) using Hyper-Dual Tensors, up to second order derivatives
- Create functions in terms of Hyper-Dual Tensors
- Evaluate the function, the gradient (jacobian) and the hessian on given input arrays
- Straight-forward definition of custom functions in variational-calculus notation
- Stable gradient and hessian of eigenvalues `eigvalsh` in case of repeated equal eigenvalues

# Not Features
- Not imitating NumPy (like [Autograd](https://github.com/HIPS/autograd))
- No arbitrary-order gradients

# Usage
Let's define a scalar-valued function which operates on a tensor.

```python
import tensortrax as tr
import tensortrax.math as tm

def fun(F):
    C = F.T() @ F
    I1 = tm.trace(C)
    J = tm.linalg.det(F)
    return J ** (-2 / 3) * I1 - 3
```

The hessian of the scalar-valued function w.r.t. the function argument is evaluated by variational calculus (Forward Mode AD implemented as Hyper-Dual Tensors). The function is called once for each component of the hessian (symmetry is taken care of). The function and the gradient are evaluated with no additional computational cost. 

```python
import numpy as np

# some random input data
np.random.seed(125161)
F = np.random.rand(3, 3, 8, 50) / 10
for a in range(3):
    F[a, a] += 1

# W = tr.function(fun, ntrax=2)(F)
# dWdF, W = tr.gradient(fun, ntrax=2)(F)
d2WdF2, dWdF, W = tr.hessian(fun, ntrax=2)(F)
```

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

The $P_{ij}$ - component of the jacobian $\boldsymbol{P}$ is now numerically evaluated by setting the respective variational component $\delta P_{ij}$ of the tensor to one and all other components to zero. In total, $i \cdot j$ function calls are necessary to assemble the full jacobian. For example, the $12$ - component is evaluated as follows:

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
from tensortrax import Tensor, f, δ, Δ, Δδ
from tensortrax.math import trace

δF_12 = np.array([
    [0, 1, 0], 
    [0, 0, 0], 
    [0, 0, 0],
], dtype=float)

ΔF_23 = np.array([
    [0, 0, 0], 
    [0, 0, 1], 
    [0, 0, 0],
], dtype=float)

x = np.eye(3) + np.arange(9).reshape(3, 3) / 10
F = Tensor(x=x, δx=δF_12, Δx=ΔF_23, Δδx=None)
I1_C = trace(F.T() @ F)
```

The function as well as the gradient and hessian components are accessible as:

```python
ψ      =  f(I1_C)
P_12   =  δ(I1_C) # (= Δ(I1_C))
A_1223 = Δδ(I1_C)
```

To obtain full gradients and hessians in one function call, `tensortrax` provides helpers (decorators) which handle the multiple function calls.

```python
# input data with 0 trailing axes
gradient(lambda F: trace(F.T() @ F), ntrax=0)(x)
hessian(lambda F: trace(F.T() @ F), ntrax=0)(x)
```
