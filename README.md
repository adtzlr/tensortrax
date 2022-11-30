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
    J = tm.det(F)
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
The calculus of variation deals with variations, i.e. small changes in functions and functionals. A small-change of a function is evaluated by taking the partial derivatives of the function w.r.t. the tensor components (the jacobian) and contracted by the variation (small change) of the function argument.

```math
\psi = \psi(\boldsymbol{E})

\delta \psi = \delta \psi(\boldsymbol{E}, \delta \boldsymbol{E})
```

Let's take the trace as an example. The variation of the trace of a tensor product is evaluated as the trace of the variation (small-change) of the tensor.

```math
\psi = trace(\boldsymbol{F}^T \boldsymbol{F}) = \boldsymbol{F} : \boldsymbol{F}

\delta \psi = \delta \boldsymbol{F} : \boldsymbol{F} + \boldsymbol{F} : \delta \boldsymbol{F} = 2 \ \boldsymbol{F} : \delta \boldsymbol{F}
```

The $P_{ij}$-component of the jacobian $\boldsymbol{P}$ is now numerically evaluated by setting the respective variation component of the tensor to one and all other components to zero. In total, $i \cdot j$ function calls are necessary to assemble the full jacobian. For example, the $11$-component is evaluated as follows:

```math
\delta \boldsymbol{F}_{(12)} = \begin{bmatrix} 0 & 1 & 0 \\ 0 & 0 & 0 \\ 0 & 0 & 0 \end{bmatrix}

\delta_{(12)} \psi = \frac{\partial \psi}{\partial F_{12} = 2 \ \boldsymbol{F} : \delta \boldsymbol{F}_{(12)} = 2 \ \boldsymbol{F} : \begin{bmatrix} 0 & 1 & 0 \\ 0 & 0 & 0 \\ 0 & 0 & 0 \end{bmatrix}
```

The second order derivative, i.e. the partial derivative of another partial derivative is evaluated by a further small-change (for a linear map, this is equal to the linearization).

```math
\Delta \delta \psi = \delta \boldsymbol{F} : \Delta \boldsymbol{F} + \Delta \boldsymbol{F} : \delta \boldsymbol{F} = 2 \ \delta \boldsymbol{F} : \Delta \boldsymbol{F} + 2 \ \boldsymbol{F} : \Delta \delta \boldsymbol{F}
```

Once again, each component $\mathbb{A}_{ijkl}$ of the fourth-order hessian $\mathbb{A}$ is numerically evaluated as shown for the component $\mathbb{A}_{1223}$.

```math
\delta \boldsymbol{F}_{(12)} = \begin{bmatrix} 0 & 1 & 0 \\ 0 & 0 & 0 \\ 0 & 0 & 0 \end{bmatrix}

\Delta \boldsymbol{F}_{(23)} = \begin{bmatrix} 0 & 0 & 0 \\ 0 & 0 & 1 \\ 0 & 0 & 0 \end{bmatrix}

\Delta_{(23}) \delta_{(12}) \psi = \delta \boldsymbol{F}_{(12)} : \Delta \boldsymbol{F}_{(23)} + \Delta \boldsymbol{F}_{(23)} : \delta \boldsymbol{F}_{(12)} = 2 \ \delta \boldsymbol{F}_{(12)} : \Delta \boldsymbol{F}_{(23)} + 2 \ \boldsymbol{F} : \Delta \delta \boldsymbol{F}
```