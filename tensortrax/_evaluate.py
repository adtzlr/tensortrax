r"""
 _                            
| |                          ████████╗██████╗  █████╗ ██╗  ██╗
| |_ ___ _ __  ___  ___  _ __╚══██╔══╝██╔══██╗██╔══██╗╚██╗██╔╝
| __/ _ \ '_ \/ __|/ _ \| '__|  ██║   ██████╔╝███████║ ╚███╔╝ 
| ||  __/ | | \__ \ (_) | |     ██║   ██╔══██╗██╔══██║ ██╔██╗ 
 \__\___|_| |_|___/\___/|_|     ██║   ██║  ██║██║  ██║██╔╝ ██╗
                                ╚═╝   ╚═╝  ╚═╝╚═╝  ╚═╝╚═╝  ╚═╝  
"""

from copy import copy
from functools import wraps
from threading import Thread

import numpy as np

from ._tensor import Tensor, f, istensor, δ
from .math._special import from_triu_1d, from_triu_2d, triu_1d


def take(fun, item=0):
    "Evaluate the function and take only the selected item."

    @wraps(fun)
    def evaluate(*args, **kwargs):
        return fun(*args, **kwargs)[item]

    return evaluate


def add_tensor(args, kwargs, wrt, δx, ntrax, sym):
    "Modify the arguments and replace the w.r.t.-argument by a tensor."

    kwargs_out = copy(kwargs)
    args_out = list(args)

    if isinstance(wrt, str):
        if sym:
            kwargs_out[wrt] = from_triu_1d(
                Tensor(x=triu_1d(kwargs[wrt]), δx=δx, ntrax=ntrax)
            )
        else:
            kwargs_out[wrt] = Tensor(x=kwargs[wrt], δx=δx, ntrax=ntrax)

    elif isinstance(wrt, int):
        if sym:
            args_out[wrt] = from_triu_1d(
                Tensor(x=triu_1d(args[wrt]), δx=δx, ntrax=ntrax)
            )
        else:
            args_out[wrt] = Tensor(x=args[wrt], δx=δx, ntrax=ntrax)

    return args_out, kwargs_out


def arg_to_tensor(args, kwargs, wrt, sym):
    "Return the argument which will be replaced by a tensor."

    if isinstance(wrt, str):
        x = kwargs[wrt] if not sym else triu_1d(kwargs[wrt])
    elif isinstance(wrt, int):
        x = args[wrt] if not sym else triu_1d(args[wrt])
    else:
        raise TypeError(f"w.r.t. {wrt} not supported.")

    return x


def function(fun, wrt=0, ntrax=0, parallel=False):
    "Evaluate a scalar-valued function."

    @wraps(fun)
    def evaluate_function(*args, **kwargs):
        args, kwargs = add_tensor(args, kwargs, wrt, None, ntrax, False)
        return fun(*args, **kwargs).x

    return evaluate_function


def asarray(A):
    asarr = lambda A: asarray(A) if istensor(A[0]) else np.asarray(A)
    return Tensor(asarr([f(Ai) for Ai in A]), asarr([δ(Ai) for Ai in A]), ntrax=A[0].ntrax)


def jacobian(fun, wrt=0, ntrax=0, parallel=False, full_output=False):
    "Evaluate the jacobian of a function."

    @wraps(fun)
    def evaluate_jacobian(*args, **kwargs):

        x = arg_to_tensor(args, kwargs, wrt, False)
        t = Tensor(x, ntrax=ntrax)
        δx = np.eye(t.size)
        indices = range(t.size)

        args0, kwargs0 = add_tensor(args, kwargs, wrt, None, ntrax, False)
        shape = fun(*args0, **kwargs0).shape

        dfdx = []

        def kernel(a, wrt, δx, ntrax, args, kwargs):
            args, kwargs = add_tensor(args, kwargs, wrt, δx[a], ntrax, False)
            dfdx.append(fun(*args, **kwargs))

        if not parallel:
            for a in indices:
                kernel(a, wrt, δx, ntrax, args, kwargs)

        else:
            threads = [
                Thread(target=kernel, args=(a, wrt, δx, ntrax, args, kwargs))
                for a in indices
            ]

            for th in threads:
                th.start()

            for th in threads:
                th.join()
        
        return asarray(dfdx).reshape(*shape, *t.shape)

    return evaluate_jacobian


def gradient(fun, wrt=0, ntrax=0, parallel=False, full_output=False, sym=False):
    "Evaluate the gradient of a scalar-valued function."

    @wraps(fun)
    def evaluate_gradient(*args, **kwargs):

        x = arg_to_tensor(args, kwargs, wrt, sym)
        t = Tensor(x, ntrax=ntrax)
        indices = range(t.size)

        fx = np.zeros((1, *t.trax))
        dfdx = np.zeros((t.size, *t.trax))
        δx = np.eye(t.size)

        if sym:
            idx_off_diag = {1: None, 3: [1], 6: [1, 2, 4]}[t.size]
            δx[idx_off_diag] /= 2

        def kernel(a, wrt, δx, ntrax, sym, args, kwargs):
            args, kwargs = add_tensor(args, kwargs, wrt, δx[a], ntrax, sym)
            func = fun(*args, **kwargs)
            fx[:] = f(func)
            dfdx[a] = δ(func)

        if not parallel:
            for a in indices:
                kernel(a, wrt, δx, ntrax, sym, args, kwargs)

        else:
            threads = [
                Thread(target=kernel, args=(a, wrt, δx, ntrax, sym, args, kwargs))
                for a in indices
            ]

            for th in threads:
                th.start()

            for th in threads:
                th.join()

        if sym:
            dfdx = from_triu_1d(dfdx)
            shape = dfdx.shape[:2]
        else:
            shape = t.shape

        if full_output:
            return np.array(dfdx).reshape(*shape, *t.trax), fx[0]
        else:
            return np.array(dfdx).reshape(*shape, *t.trax)

    return evaluate_gradient


def gradient_vector_product(fun, wrt=0, ntrax=0, parallel=False):
    "Evaluate the gradient-vector-product of a function."

    @wraps(fun)
    def evaluate_gradient_vector_product(*args, δx, **kwargs):
        args, kwargs = add_tensor(args, kwargs, wrt, δx, ntrax, False)
        return fun(*args, **kwargs).δx

    return evaluate_gradient_vector_product
