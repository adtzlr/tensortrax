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

import tensortrax as tr
import tensortrax.math as tm


def fun_tensortrax(C):
    return tm.trace(C) - tm.log(tm.linalg.det(C))


def fun_autograd(C):
    Csym = (anp.einsum("ij...->ji...", C) + C) / 2
    return anp.trace(Csym) - anp.log(anp.linalg.det(Csym.T).T)


def pre_tensortrax(n, **kwargs):
    np.random.seed(4519245)
    F = np.eye(3).reshape(3, 3, 1) + (np.random.rand(3, 3, n) - 0.5) / 10
    C = np.einsum("ki...,kj...->ij...", F, F)
    stress = tr.gradient(fun_tensortrax, **kwargs)
    elasticity = tr.hessian(fun_tensortrax, **kwargs)
    return C, stress, elasticity


def pre_autograd(n, **kwargs):
    np.random.seed(4519245)
    F = anp.eye(3).reshape(3, 3, 1) + (anp.random.rand(3, 3, n) - 0.5) / 10
    C = anp.einsum("ki...,kj...->ij...", F, F)
    reduce = lambda fun: lambda C: anp.sum(fun(C), -1)
    stress = jacobian(reduce(fun_autograd))
    elasticity = jacobian(reduce(jacobian(reduce(fun_autograd))))
    return C, stress, elasticity


tensors = 2 ** np.arange(0, 21, 2)
time_gradient_tensortrax = []
time_hessian_tensortrax = []
time_gradient_autograd = []
time_hessian_autograd = []

kwargs = dict(ntrax=1, sym=True)
number = 3

print("Tensortrax Benchmark (Comparison with Autograd)")
print("===============================================")
print("")
print(f"tensortrax {tr.__version__}")
print("")

for i, n in enumerate(tensors):
    c, stress, elasticity = pre_tensortrax(n, **kwargs)
    C, Stress, Elasticity = pre_autograd(n, **kwargs)

    s = stress(c)
    e = elasticity(c)

    S = Stress(C)
    E = Elasticity(C)

    assert np.allclose(s, S)
    assert np.allclose(e, E)

    time_gradient_tensortrax.append(timeit(lambda: stress(c), number=number) / number)
    time_hessian_tensortrax.append(
        timeit(lambda: elasticity(c), number=number) / number
    )
    time_gradient_autograd.append(timeit(lambda: Stress(C), number=number) / number)
    time_hessian_autograd.append(timeit(lambda: Elasticity(C), number=number) / number)

    print(f"...Evaluate timings... {i+1}/{len(tensors)}")

print("")
print("|         | (Tensortrax)  |  (Autograd)   |         |")
print("| Tensors | Gradient in s | Gradient in s | Speedup |")
print("| ------- | ------------- | ------------- | ------- |")
for n, t_grad_trax, t_grad_autograd in zip(
    tensors, time_gradient_tensortrax, time_gradient_autograd
):
    speedup = t_grad_autograd / t_grad_trax
    print(
        f"| {n:7d} | {t_grad_trax:13.5f} | {t_grad_autograd:13.5f} | x{speedup:6.2f} |"
    )

print("")
print("")
print("|         | (Tensortrax)  |  (Autograd)   |         |")
print("| Tensors | Hessian in s  | Hessian in s  | Speedup |")
print("| ------- | ------------- | ------------- | ------- |")
for n, t_hess_trax, t_hess_autograd in zip(
    tensors, time_hessian_tensortrax, time_hessian_autograd
):
    speedup = t_hess_autograd / t_hess_trax
    print(
        f"| {n:7d} | {t_hess_trax:13.5f} | {t_hess_autograd:13.5f} | x{speedup:6.2f} |"
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
plt.xlabel(r"Number of input tensors $\longrightarrow$")
plt.ylabel(r"Runtime in s $\longrightarrow$")
plt.legend()
plt.tight_layout()
plt.savefig("benchmark_tensortrax_vs_autograd.svg", transparent=False)
plt.savefig("benchmark_tensortrax_vs_autograd.png", transparent=False)
