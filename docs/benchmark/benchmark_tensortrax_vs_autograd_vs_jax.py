r"""
 _                            
| |                          ████████╗██████╗  █████╗ ██╗  ██╗
| |_ ___ _ __  ___  ___  _ __╚══██╔══╝██╔══██╗██╔══██╗╚██╗██╔╝
| __/ _ \ '_ \/ __|/ _ \| '__|  ██║   ██████╔╝███████║ ╚███╔╝ 
| ||  __/ | | \__ \ (_) | |     ██║   ██╔══██╗██╔══██║ ██╔██╗ 
 \__\___|_| |_|___/\___/|_|     ██║   ██║  ██║██║  ██║██╔╝ ██╗
                                ╚═╝   ╚═╝  ╚═╝╚═╝  ╚═╝╚═╝  ╚═╝  
"""

from timeit import timeit

import matplotlib.pyplot as plt
import numpy as np
from autograd import jacobian
from autograd import numpy as anp
from jax import jacobian as jjacobian
from jax import numpy as jnp, jit

import tensortrax as tr
import tensortrax.math as tm


def fun_tensortrax(C):
    return tm.trace(C) - tm.log(tm.linalg.det(C))


def det(A):
    return (
        A[0, 0] * A[1, 1] * A[2, 2]
        + A[0, 1] * A[1, 2] * A[2, 0]
        + A[0, 2] * A[1, 0] * A[2, 1]
        - A[2, 0] * A[1, 1] * A[0, 2]
        - A[2, 1] * A[1, 2] * A[0, 0]
        - A[2, 2] * A[1, 0] * A[0, 1]
    )


def fun_autograd(C):
    Csym = (anp.einsum("ij...->ji...", C) + C) / 2
    return anp.trace(Csym) - anp.log(det(Csym))


@jit
def fun_jax(C):
    Csym = (jnp.einsum("ij...->ji...", C) + C) / 2
    return jnp.trace(Csym) - jnp.log(det(Csym))


def pre_tensor(n):
    np.random.seed(4519245)
    F = np.eye(3).reshape(3, 3, 1) + (np.random.rand(3, 3, n) - 0.5) / 10
    C = np.einsum("ki...,kj...->ij...", F, F)
    return C


def pre_tensortrax(**kwargs):
    stress = tr.gradient(fun_tensortrax, **kwargs)
    elasticity = tr.hessian(fun_tensortrax, **kwargs)
    return stress, elasticity


def pre_autograd(**kwargs):
    reduce = lambda fun: lambda C: anp.sum(fun(C), axis=-1)
    stress = jacobian(reduce(fun_autograd))
    elasticity = jacobian(reduce(jacobian(reduce(fun_autograd))))
    return stress, elasticity


def pre_jax(**kwargs):
    reduce = lambda fun: lambda C: jnp.sum(fun(C), axis=-1)
    stress = jjacobian(reduce(fun_jax))
    elasticity = jjacobian(reduce(jjacobian(reduce(fun_jax))))
    return stress, elasticity


tensors = 2 ** np.arange(0, 21, 2)
time_gradient_tensortrax = []
time_hessian_tensortrax = []
time_gradient_autograd = []
time_hessian_autograd = []
time_gradient_jax = []
time_hessian_jax = []

kwargs = dict(ntrax=1, sym=True, parallel=False)
number = 1

print("Tensortrax Benchmark (Comparison with Autograd/JAX)")
print("===================================================")
print("")
print(f"tensortrax {tr.__version__}")
print("")

for i, n in enumerate(tensors):
    C = pre_tensor(n)
    stress, elasticity = pre_tensortrax(**kwargs)
    Stress, Elasticity = pre_autograd(**kwargs)
    JStress, JElasticity = pre_jax(**kwargs)

    JStress(C)
    JElasticity(C)

    if n < 10000:
        s = stress(C)
        e = elasticity(C)

        S = Stress(C)
        E = Elasticity(C)

        assert np.allclose(s, S)
        assert np.allclose(e, E)

        del s
        del e
        del S
        del E

    time_gradient_tensortrax.append(timeit(lambda: stress(C), number=number) / number)
    time_hessian_tensortrax.append(
        timeit(lambda: elasticity(C), number=number) / number
    )
    time_gradient_autograd.append(timeit(lambda: Stress(C), number=number) / number)
    time_hessian_autograd.append(timeit(lambda: Elasticity(C), number=number) / number)
    time_gradient_jax.append(timeit(lambda: JStress(C), number=number) / number)
    time_hessian_jax.append(timeit(lambda: JElasticity(C), number=number) / number)

    print(f"...Evaluate timings... {i+1}/{len(tensors)}")

print("")
print(
    "| Tensors | Gradient (Tensortrax) in s | Gradient (Autograd) in s | Speedup | Gradient (JAX) in s | Speedup |"
)
print(
    "| ------- | -------------------------- | ------------------------ | ------- | ------------------- | ------- |"
)
for n, t_grad_trax, t_grad_autograd, t_grad_jax in zip(
    tensors, time_gradient_tensortrax, time_gradient_autograd, time_gradient_jax
):
    speedup = t_grad_autograd / t_grad_trax
    speedup_jax = t_grad_jax / t_grad_trax
    print(
        f"| {n:7d} | {t_grad_trax:26.5f} | {t_grad_autograd:24.5f} | x{speedup:6.2f} | {t_grad_jax:19.5f} | x{speedup_jax:6.2f} |"
    )

print("")
print("")
print(
    "| Tensors | Hessian (Tensortrax) in s  | Hessian (Autograd) in s  | Speedup | Hessian (JAX) in s  | Speedup |"
)
print(
    "| ------- | -------------------------- | ------------------------ | ------- | ------------------- | ------- |"
)
for n, t_hess_trax, t_hess_autograd, t_hess_jax in zip(
    tensors, time_hessian_tensortrax, time_hessian_autograd, time_hessian_jax
):
    speedup = t_hess_autograd / t_hess_trax
    speedup_jax = t_hess_jax / t_hess_trax
    print(
        f"| {n:7d} | {t_hess_trax:26.5f} | {t_hess_autograd:24.5f} | x{speedup:6.2f} | {t_hess_jax:19.5f} | x{speedup_jax:6.2f} |"
    )


plt.figure()
plt.title(r"Strain Energy Function $\psi(C) = \mathrm{tr}(C) - \ln(\det(C))$")
plt.loglog(
    tensors,
    time_gradient_tensortrax,
    "C0",
    label="Gradient (Tensortrax) $\partial \psi~/~\partial C$",
)
plt.loglog(
    tensors,
    time_gradient_autograd,
    "C0--",
    label="Gradient (Autograd) $\partial \psi~/~\partial C$",
)
plt.loglog(
    tensors,
    time_gradient_jax,
    "C0:",
    label="Gradient (JAX, CPU) $\partial \psi~/~\partial C$",
)
plt.loglog(
    tensors,
    time_hessian_tensortrax,
    "C1",
    label="Hessian (Tensortrax) $\partial^2 \psi~/~\partial C \partial C$",
)
plt.loglog(
    tensors,
    time_hessian_autograd,
    "C1--",
    label="Hessian (Autograd) $\partial^2 \psi~/~\partial C \partial C$",
)
plt.loglog(
    tensors,
    time_hessian_jax,
    "C1:",
    label="Hessian (JAX, CPU) $\partial^2 \psi~/~\partial C \partial C$",
)
plt.xlabel(r"Number of input tensors $\longrightarrow$")
plt.ylabel(r"Runtime in s $\longrightarrow$")
plt.legend()
plt.tight_layout()
plt.savefig("benchmark_tensortrax_vs_autograd_vs_jax.svg", transparent=False)
plt.savefig("benchmark_tensortrax_vs_autograd_vs_jax.png", transparent=False)
