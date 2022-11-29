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

# Features
- Designed to operate on input arrays with trailing axes
- Essential vector/tensor dual-number math, including limited support for `einsum` (restricted to max. two operands)
- Forward Mode Automatic Differentiation (AD)
- Create functions in terms of Hyper-Dual Tensors
- Evaluate the function, the gradient (jacobian) and the hessian on given input arrays

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

The hessian w.r.t. the function argument is evaluated by Forward Mode Automatic Differentiation (AD).

```python
import numpy as np

# some random input data
np.random.seed(125161)
F = np.random.rand(3, 3, 8, 50) / 10
for a in range(3):
    F[a, a] += 1

W = tr.function(fun)(F)
dWdF, W = tr.gradient(fun)(F)
d2WdF2, dWdF, W = tr.hessian(fun)(F)
```