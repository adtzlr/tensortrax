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

# Highlights

- Designed to operate on input arrays with (elementwise-operating) trailing axes
- Essential vector/tensor Hyper-Dual number math, including limited support for `einsum` (restricted to max. three operands)
- Math is limited but similar to NumPy, try to use `import tensortrax.math as tm` instead of `import numpy as np` inside functions to be differentiated
- Forward Mode Automatic Differentiation (AD) using Hyper-Dual Tensors, up to second order derivatives
- Create functions in terms of Hyper-Dual Tensors
- Evaluate the function, the gradient (jacobian) and the hessian of scalar-valued functions or functionals on given input arrays
- Straight-forward definition of custom functions in variational-calculus notation
- Stable gradient and hessian of eigenvalues obtained from `eigvalsh` in case of repeated equal eigenvalues

Please keep in mind that `tensortrax` is not imitating a 100% full-featured NumPy, e.g. like [Autograd](https://github.com/HIPS/autograd) [[1]](https://github.com/HIPS/autograd). No arbitrary-order gradients or gradients-of-gradients are supported. The capability is limited to first- and second order gradients of a given function. Also, `tensortrax` provides no support for `dtype=complex`.

# Motivation
Gradient and hessian evaluations of functions or functionals based on tensor-valued input arguments are a fundamental repetitive and (error-prone) task in constitutive hyperelastic material formulations used in continuum mechanics of solid bodies. In the worst case, conceptual ideas are impossible to pursue because the required tensorial derivatives are not readily achievable. The Hyper-Dual number approach enables a generalized and systematic way to overcome this deficiency [[2]](https://arc.aiaa.org/doi/abs/10.2514/6.2011-886). Compared to existing Hyper-Dual Number libaries ([[3]](https://github.com/itt-ustutt/num-dual), [[4]](https://github.com/oberbichler/HyperJet)) which introduce a new (hyper-) dual `dtype` (treated as `dtype=object` in NumPy), `tensortrax` relies on its own `Tensor` class. This approach involves a re-definition of all essential math operations (and NumPy-functions), whereas the `dtype`-approach supports most basic math operations out of the box. However, in `tensortrax` NumPy and all its underlying linear algebra functions operate on default data types (e.g. `dtype=float`). This allows to support functions like `np.einsum()`. Beside the differences concerning the underlying `dtype`, `tensortrax` is formulated on easy-to-understand (tensorial) calculus of variation. Hence, gradient- and hessian-vector products are evaluated with very little overhead compared to analytic formulations.

# Usage
Let's define a scalar-valued function which operates on a tensor. The math module `tensortrax.math` provides some essential NumPy-like functions including linear algebra.

```python
import tensortrax as tr
import tensortrax.math as tm


def fun(F, mu=1):
    C = F.T() @ F
    I1 = tm.trace(C)
    J = tm.linalg.det(F)
    return mu / 2 * (J ** (-2 / 3) * I1 - 3)
```

The Hessian of the scalar-valued function w.r.t. the chosen function argument (here, `wrt=0` or `wrt="F"`) is evaluated by variational calculus (Forward Mode AD implemented as Hyper-Dual Tensors). The function is called once for each component of the hessian (symmetry is taken care of). The function and the gradient are evaluated with no additional computational cost. Optionally, the function, gradient and Hessian calls are executed in parallel (threaded).

```python
import numpy as np

# some random input data
np.random.seed(125161)
F = (np.eye(3) + np.random.rand(50, 8, 3, 3) / 10).T

# W = tr.function(fun, wrt=0, ntrax=2)(F)
# dWdF = tr.gradient(fun, wrt=0, ntrax=2)(F)
# d2WdF2, dWdF, W = tr.hessian(fun, wrt="F", ntrax=2, full_output=True)(F=F)
d2WdF2 = tr.hessian(fun, wrt="F", ntrax=2, parallel=False)(F=F)
```

Another possibility is to define and operate on Tensors manually. This enables more flexible coding, which wouldn't be possible with the builtin functions. The Hu-Washizu Three-Field-Variational principle for nearly incompressible hyperelastic solids [[5]](https://doi.org/10.1017/CBO9780511755446) is used here to obtain mixed partial derivatives. Some random input arrays are generated and a Tensor is created for each variable. After performing some math, the hessian of the resulting tensor object is extracted.

```python
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
    C = F.T() @ F
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
```

In a similar way, the gradient may be obtained by initiating a Tensor with the gradient argument.

```python
# init Tensors to be used with first partial derivatives
F.init(gradient=True, δx=False)
p.init(gradient=True, δx=True)
J.init(gradient=True, δx=False)

# evaluate a partial derivative
dWdp = tr.δ(W(F, p, J))
```

# Performance
A [benchmark](https://github.com/adtzlr/tensortrax/blob/main/docs/benchmark/benchmark_tensortrax_vs_autograd.py) for the gradient and hessian runtimes of an isotropic hyperelastic strain energy function demonstrates the performance of this package compared to [Autograd](https://github.com/HIPS/autograd) [[1]](https://github.com/HIPS/autograd). The hessian is evaluated in about 2.5 seconds for one million input tensors (Intel Core i7-11850H, 32GB RAM). While the runtimes of the gradients evaluated by Tensortrax are similar compared to those of Autograd, the Hessian evaluation in Tensortrax is **much faster** than in Autograd (see **Table 1**, **Table 2** and **Figure 1**). Note that this is only one exemplaric benchmark for a specific kind of function - runtimes may vary.

```math
\psi(\boldsymbol{C}) = tr(\boldsymbol{C}) - \ln(\det(\boldsymbol{C}))
```

**Table 1**: *Runtimes of Gradients evaluated by `tensortrax` and `autograd`.*

| Tensors | Gradient (Tensortrax) in s | Gradient (Autograd) in s | Speedup |
| ------- | -------------------------- | ------------------------ | ------- |
|       1 |                    0.00071 |                  0.00082 | x  1.16 |
|       4 |                    0.00075 |                  0.00070 | x  0.93 |
|      16 |                    0.00075 |                  0.00071 | x  0.94 |
|      64 |                    0.00083 |                  0.00077 | x  0.94 |
|     256 |                    0.00097 |                  0.00094 | x  0.97 |
|    1024 |                    0.00160 |                  0.00161 | x  1.01 |
|    4096 |                    0.00568 |                  0.00428 | x  0.75 |
|   16384 |                    0.01233 |                  0.01690 | x  1.37 |
|   65536 |                    0.06103 |                  0.06756 | x  1.11 |
|  262144 |                    0.27910 |                  0.26911 | x  0.96 |
| 1048576 |                    1.15561 |                  1.09512 | x  0.95 |


**Table 2**: *Runtimes of Hessians evaluated by `tensortrax` and `autograd`.*

| Tensors | Hessian (Tensortrax) in s  | Hessian (Autograd) in s  | Speedup |
| ------- | -------------------------- | ------------------------ | ------- |
|       1 |                    0.00068 |                  0.01348 | x 19.77 |
|       4 |                    0.00073 |                  0.00723 | x  9.87 |
|      16 |                    0.00076 |                  0.00718 | x  9.41 |
|      64 |                    0.00083 |                  0.00792 | x  9.58 |
|     256 |                    0.00111 |                  0.01044 | x  9.43 |
|    1024 |                    0.00214 |                  0.02094 | x  9.79 |
|    4096 |                    0.00844 |                  0.06148 | x  7.28 |
|   16384 |                    0.02544 |                  0.23778 | x  9.35 |
|   65536 |                    0.11685 |                  0.95208 | x  8.15 |
|  262144 |                    0.51561 |                  3.94024 | x  7.64 |
| 1048576 |                    2.13629 |                 16.15154 | x  7.56 |

![benchmark_tensortrax_vs_autograd](https://user-images.githubusercontent.com/5793153/217664124-260514d0-854e-42f1-98ed-68bdbfafaa5b.svg)

**Figure 1**: *Runtime vs. Number of input tensors - plot for Gradients and Hessians evaluated by `tensortrax` and `autograd`.*

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

> **Hint**: *Feel free to [contribute](https://github.com/adtzlr/tensortrax/fork) missing math-functions to [`src/tensortrax/math/_math_tensor.py`](https://github.com/adtzlr/tensortrax/blob/main/src/tensortrax/math/_math_tensor.py)* :page_with_curl: :pencil2:.

# References
1. D. Maclaurin, D. Duvenaud, M. Johnson and J. Townsend, *Autograd*. [Online]. Available: https://github.com/HIPS/autograd.
2. J. Fike and J. Alonso, *The Development of Hyper-Dual Numbers for Exact Second-Derivative Calculations*, 49th AIAA Aerospace Sciences Meeting including the New Horizons Forum and Aerospace Exposition. American Institute of Aeronautics and Astronautics, Jan. 04, 2011, [![10.2514/6.2011-886](https://zenodo.org/badge/DOI/10.2514/6.2011-886.svg)](https://doi.org/10.2514/6.2011-886).
3. P. Rehner and G. Bauer, *Application of Generalized (Hyper-) Dual Numbers in Equation of State Modeling*, Frontiers in Chemical Engineering, vol. 3, 2021. Available: https://github.com/itt-ustutt/num-dual.
4. T. Oberbichler, *HyperJet*. [Online]. Available: http://github.com/oberbichler/HyperJet.
5. J. Bonet and R. D. Wood, *Nonlinear Continuum Mechanics for Finite Element Analysis*, 2nd ed. Cambridge: Cambridge University Press, 2008, [![10.1017/CBO9780511755446](https://zenodo.org/badge/DOI/10.1017/CBO9780511755446.svg)](https://doi.org/10.1017/CBO9780511755446).
