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

import tensortrax as tr
import tensortrax.math as tm


def neo_hooke(C):
    return tm.trace(C) - tm.log(tm.linalg.det(C))


def pre(n, **kwargs):
    np.random.seed(4519245)
    F = np.eye(3).reshape(3, 3, 1) + (np.random.rand(3, 3, n) - 0.5) / 10
    C = np.einsum("ki...,kj...->ij...", F, F)
    stress = tr.gradient(neo_hooke, **kwargs)
    elasticity = tr.hessian(neo_hooke, **kwargs)
    return C, stress, elasticity


tensors = 2 ** np.arange(0, 21, 2)
time_gradient = []
time_hessian = []

print("")
print("| Tensors | Gradient in s | Hessian in s |")
print("| ------- | ------------- | ------------ |")

kwargs = dict(ntrax=1, sym=True, parallel=False)
number = 3

for n in tensors:
    C, stress, elasticity = pre(n, **kwargs)
    stress(C), elasticity(C)
    time_gradient.append(timeit(lambda: stress(C), number=number) / number)
    time_hessian.append(timeit(lambda: elasticity(C), number=number) / number)
    print(f"| {n:7d} | {time_gradient[-1]:13.5f} | {time_hessian[-1]:12.5f} |")

plt.figure()
plt.title(r"Strain Energy Function $\psi(C) = \mathrm{tr}(C) - \ln(\det(C))$")
plt.loglog(tensors, time_gradient, "C0", label=r"Gradient $\partial \psi~/~\partial C$")
plt.loglog(
    tensors,
    time_hessian,
    "C1",
    label=r"Hessian $\partial^2 \psi~/~\partial C \partial C$",
)
plt.xlabel(r"Number of input tensors $\longrightarrow$")
plt.ylabel(r"Runtime in s $\longrightarrow$")
plt.legend()
plt.tight_layout()
plt.savefig("benchmark.svg", transparent=False)
plt.savefig("benchmark.png", transparent=False)
