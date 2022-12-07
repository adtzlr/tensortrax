r"""
 _                            
| |                          ████████╗██████╗  █████╗ ██╗  ██╗
| |_ ___ _ __  ___  ___  _ __╚══██╔══╝██╔══██╗██╔══██╗╚██╗██╔╝
| __/ _ \ '_ \/ __|/ _ \| '__|  ██║   ██████╔╝███████║ ╚███╔╝ 
| ||  __/ | | \__ \ (_) | |     ██║   ██╔══██╗██╔══██║ ██╔██╗ 
 \__\___|_| |_|___/\___/|_|     ██║   ██║  ██║██║  ██║██╔╝ ██╗
                                ╚═╝   ╚═╝  ╚═╝╚═╝  ╚═╝╚═╝  ╚═╝  
"""

from threading import Thread
import numpy as np

from ._tensor import Tensor, f, δ, Δδ


def function(fun, ntrax=0, parallel=False):
    "Evaluate a scalar-valued function."

    def evaluate_function(x, *args, **kwargs):
        return fun(Tensor(x, ntrax=ntrax), *args, **kwargs).x

    return evaluate_function


def gradient(fun, ntrax=0, parallel=False):
    "Evaluate the gradient of a scalar-valued function."

    def evaluate_gradient(x, *args, **kwargs):

        t = Tensor(x, ntrax=ntrax)
        indices = range(t.size)

        fx = np.zeros((1, *t.trax))
        dfdx = np.zeros((t.size, *t.trax))
        δx = Δx = np.eye(t.size)

        def kernel(a, x, δx, Δx, args, kwargs):
            t = Tensor(x, δx=δx[a], Δx=Δx[a], ntrax=ntrax)
            func = fun(t, *args, **kwargs)
            fx[:] = f(func)
            dfdx[a] = δ(func)

        if not parallel:
            for a in indices:
                kernel(a, x, δx, Δx, args, kwargs)

        else:
            threads = [
                Thread(target=kernel, args=(a, x, δx, Δx, args, kwargs))
                for a in indices
            ]

            for th in threads:
                th.start()

            for th in threads:
                th.join()

        return np.array(dfdx).reshape(*t.shape, *t.trax), fx[0]

    return evaluate_gradient


def hessian(fun, ntrax=0, parallel=False):
    "Evaluate the hessian of a scalar-valued function."

    def evaluate_hessian(x, *args, **kwargs):

        t = Tensor(x, ntrax=ntrax)
        indices = np.array(np.triu_indices(t.size)).T

        fx = np.zeros((1, *t.trax))
        dfdx = np.zeros((t.size, *t.trax))
        d2fdx2 = np.zeros((t.size, t.size, *t.trax))
        δx = Δx = np.eye(t.size)

        def kernel(a, b, x, δx, Δx, args, kwargs):
            t = Tensor(x, δx=δx[a], Δx=Δx[b], ntrax=ntrax)
            func = fun(t, *args, **kwargs)
            fx[:] = f(func)
            dfdx[a] = δ(func)
            d2fdx2[a, b] = d2fdx2[b, a] = Δδ(func)

        if not parallel:
            for a, b in indices:
                kernel(a, b, x, δx, Δx, args, kwargs)

        else:
            threads = [
                Thread(target=kernel, args=(a, b, x, δx, Δx, args, kwargs))
                for a, b in indices
            ]

            for th in threads:
                th.start()

            for th in threads:
                th.join()

        return (
            np.array(d2fdx2).reshape(*t.shape, *t.shape, *t.trax),
            np.array(dfdx).reshape(*t.shape, *t.trax),
            fx[0],
        )

    return evaluate_hessian


def gradient_vector_product(fun, ntrax=0, parallel=False):
    "Evaluate the gradient-vector-product of a function."

    def evaluate_gradient_vector_product(x, δx, *args, **kwargs):
        return fun(Tensor(x, δx, ntrax=ntrax), *args, **kwargs).δx

    return evaluate_gradient_vector_product


def hessian_vector_product(fun, ntrax=0, parallel=False):
    "Evaluate the gradient-vector-product of a function."

    def evaluate_hessian_vector_product(x, δx, Δx, *args, **kwargs):
        return fun(Tensor(x, δx, Δx, ntrax=ntrax), *args, **kwargs).Δδx

    return evaluate_hessian_vector_product
