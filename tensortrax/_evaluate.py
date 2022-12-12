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
from copy import copy
import numpy as np

from ._tensor import Tensor, f, δ, Δδ


def add_tensor(args, kwargs, wrt, δx, Δx, ntrax):
    "Modify the arguments and replace the w.r.t.-argument by a tensor."

    kwargs_out = copy(kwargs)
    args_out = list(args)

    if isinstance(wrt, str):
        kwargs_out[wrt] = Tensor(x=kwargs[wrt], δx=δx, Δx=Δx, ntrax=ntrax)

    elif isinstance(wrt, int):
        args_out[wrt] = Tensor(x=args[wrt], δx=δx, Δx=Δx, ntrax=ntrax)

    return args_out, kwargs_out


def arg_to_tensor(args, kwargs, wrt):
    "Return the argument which will be replaced by a tensor."

    if isinstance(wrt, str):
        x = kwargs[wrt]
    elif isinstance(wrt, int):
        x = args[wrt]
    else:
        raise TypeError(f"w.r.t. {wrt} not supported.")

    return x


def function(fun, wrt=0, ntrax=0, parallel=False):
    "Evaluate a scalar-valued function."

    def evaluate_function(*args, **kwargs):
        args, kwargs = add_tensor(args, kwargs, wrt, None, None, ntrax)
        return fun(*args, **kwargs).x

    return evaluate_function


def jacobian(fun, wrt=0, ntrax=0, parallel=False, full_output=False):
    "Evaluate the jacobian of a function."

    def evaluate_jacobian(*args, **kwargs):

        x = arg_to_tensor(args, kwargs, wrt)
        t = Tensor(x, ntrax=ntrax)
        δx = Δx = np.eye(t.size)
        indices = range(t.size)

        args0, kwargs0 = add_tensor(args, kwargs, wrt, None, None, ntrax)
        shape = fun(*args0, **kwargs0).shape
        axes = tuple([slice(None)] * len(shape))

        fx = np.zeros((*shape, *t.trax))
        dfdx = np.zeros((*shape, t.size, *t.trax))

        def kernel(a, wrt, δx, Δx, ntrax, args, kwargs):
            args, kwargs = add_tensor(args, kwargs, wrt, δx[a], Δx[a], ntrax)
            func = fun(*args, **kwargs)
            fx[axes] = f(func)
            dfdx[(*axes, a)] = δ(func)

        if not parallel:
            for a in indices:
                kernel(a, wrt, δx, Δx, ntrax, args, kwargs)

        else:
            threads = [
                Thread(target=kernel, args=(a, wrt, δx, Δx, ntrax, args, kwargs))
                for a in indices
            ]

            for th in threads:
                th.start()

            for th in threads:
                th.join()

        if full_output:
            return np.array(dfdx).reshape(*shape, *t.shape, *t.trax), fx
        else:
            return np.array(dfdx).reshape(*shape, *t.shape, *t.trax)

    return evaluate_jacobian


def gradient(fun, wrt=0, ntrax=0, parallel=False, full_output=False):
    "Evaluate the gradient of a scalar-valued function."

    def evaluate_gradient(*args, **kwargs):

        x = arg_to_tensor(args, kwargs, wrt)
        t = Tensor(x, ntrax=ntrax)
        indices = range(t.size)

        fx = np.zeros((1, *t.trax))
        dfdx = np.zeros((t.size, *t.trax))
        δx = Δx = np.eye(t.size)

        def kernel(a, wrt, δx, Δx, ntrax, args, kwargs):
            args, kwargs = add_tensor(args, kwargs, wrt, δx[a], Δx[a], ntrax)
            func = fun(*args, **kwargs)
            fx[:] = f(func)
            dfdx[a] = δ(func)

        if not parallel:
            for a in indices:
                kernel(a, wrt, δx, Δx, ntrax, args, kwargs)

        else:
            threads = [
                Thread(target=kernel, args=(a, wrt, δx, Δx, ntrax, args, kwargs))
                for a in indices
            ]

            for th in threads:
                th.start()

            for th in threads:
                th.join()

        if full_output:
            return np.array(dfdx).reshape(*t.shape, *t.trax), fx[0]
        else:
            return np.array(dfdx).reshape(*t.shape, *t.trax)

    return evaluate_gradient


def hessian(fun, wrt=0, ntrax=0, parallel=False, full_output=False):
    "Evaluate the hessian of a scalar-valued function."

    def evaluate_hessian(*args, **kwargs):

        x = arg_to_tensor(args, kwargs, wrt)
        t = Tensor(x, ntrax=ntrax)
        indices = np.array(np.triu_indices(t.size)).T

        fx = np.zeros((1, *t.trax))
        dfdx = np.zeros((t.size, *t.trax))
        d2fdx2 = np.zeros((t.size, t.size, *t.trax))
        δx = Δx = np.eye(t.size)

        def kernel(a, b, wrt, δx, Δx, ntrax, args, kwargs):
            args, kwargs = add_tensor(args, kwargs, wrt, δx[a], Δx[b], ntrax)
            func = fun(*args, **kwargs)
            fx[:] = f(func)
            dfdx[a] = δ(func)
            d2fdx2[a, b] = d2fdx2[b, a] = Δδ(func)

        if not parallel:
            for a, b in indices:
                kernel(a, b, wrt, δx, Δx, ntrax, args, kwargs)

        else:
            threads = [
                Thread(target=kernel, args=(a, b, wrt, δx, Δx, ntrax, args, kwargs))
                for a, b in indices
            ]

            for th in threads:
                th.start()

            for th in threads:
                th.join()

        if full_output:
            return (
                np.array(d2fdx2).reshape(*t.shape, *t.shape, *t.trax),
                np.array(dfdx).reshape(*t.shape, *t.trax),
                fx[0],
            )
        else:
            return np.array(d2fdx2).reshape(*t.shape, *t.shape, *t.trax)

    return evaluate_hessian


def gradient_vector_product(fun, wrt=0, ntrax=0, parallel=False):
    "Evaluate the gradient-vector-product of a function."

    def evaluate_gradient_vector_product(*args, δx, **kwargs):
        args, kwargs = add_tensor(args, kwargs, wrt, δx, None, ntrax)
        return fun(*args, **kwargs).δx

    return evaluate_gradient_vector_product


def hessian_vector_product(fun, wrt=0, ntrax=0, parallel=False):
    "Evaluate the gradient-vector-product of a function."

    def evaluate_hessian_vector_product(*args, δx, Δx, **kwargs):
        args, kwargs = add_tensor(args, kwargs, wrt, δx, Δx, ntrax)
        return fun(*args, **kwargs).Δδx

    return evaluate_hessian_vector_product
