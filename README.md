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

[![PyPI version shields.io](https://img.shields.io/pypi/v/tensortrax.svg)](https://pypi.python.org/pypi/tensortrax/) [![License: GPL v3](https://img.shields.io/badge/License-GPLv3-blue.svg)](https://www.gnu.org/licenses/gpl-3.0) ![Made with love in Graz (Austria)](https://img.shields.io/badge/Made%20with%20%E2%9D%A4%EF%B8%8F%20in-Graz%20(Austria)-0c674a) [![codecov](https://codecov.io/github/adtzlr/tensortrax/branch/main/graph/badge.svg?token=7DTH0HKYO9)](https://codecov.io/github/adtzlr/tensortrax)

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

def neo_hooke(F):
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
