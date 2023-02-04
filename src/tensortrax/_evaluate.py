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

from ._tensor import Tensor, Δδ, f, δ
from .math._special import from_triu_1d, from_triu_2d, triu_1d


def take(fun, item=0):
    "Evaluate the function and take only the selected item."

    @wraps(fun)
    def evaluate(*args, **kwargs):
        return fun(*args, **kwargs)[item]

    return evaluate


def add_tensor(
    args, kwargs, wrt, ntrax, sym=False, gradient=False, hessian=False, δx=None, Δx=None
):
    "Modify the arguments and replace the w.r.t.-argument by a tensor."

    kwargs_out = copy(kwargs)
    args_out = list(args)

    if isinstance(wrt, str):
        args_old = kwargs
        args_new = kwargs_out
    elif isinstance(wrt, int):
        args_old = args
        args_new = args_out
    else:
        raise TypeError(
            f"Type of wrt not supported. type(wrt) is {type(wrt)} (must be str or int)."
        )

    x = args_old[wrt]

    if sym:
        x = x = triu_1d(x)

    tensor = Tensor(x=x, ntrax=ntrax)
    trax = tensor.trax

    tensor.init(gradient=gradient, hessian=hessian, sym=sym, δx=δx, Δx=Δx)

    if sym:
        tensor = from_triu_1d(tensor)

    args_new[wrt] = tensor

    return args_out, kwargs_out, tensor.shape, trax


def function(fun, wrt=0, ntrax=0, parallel=False):
    "Evaluate a scalar-valued function."

    @wraps(fun)
    def evaluate_function(*args, **kwargs):
        args, kwargs, shape, trax = add_tensor(
            args, kwargs, wrt, ntrax, False, False, False
        )
        func = fun(*args, **kwargs)
        return f(func)

    return evaluate_function


def gradient(fun, wrt=0, ntrax=0, parallel=False, full_output=False, sym=False):
    "Evaluate a scalar-valued function."

    @wraps(fun)
    def evaluate_gradient(*args, **kwargs):
        args, kwargs, shape, trax = add_tensor(
            args, kwargs, wrt, ntrax, sym, True, False
        )
        func = fun(*args, **kwargs)
        grad = δ(func) if sym is False else from_triu_1d(δ(func))

        if full_output:
            trax = (1,) if len(trax) == 0 else trax
            return grad, f(func).reshape(*trax)
        else:
            return grad

    return evaluate_gradient


def hessian(fun, wrt=0, ntrax=0, parallel=False, full_output=False, sym=False):
    "Evaluate a scalar-valued function."

    @wraps(fun)
    def evaluate_hessian(*args, **kwargs):
        args, kwargs, shape, trax = add_tensor(
            args, kwargs, wrt, ntrax, sym, False, True
        )
        func = fun(*args, **kwargs)
        hess = Δδ(func) if sym is False else from_triu_2d(Δδ(func))

        if full_output:
            grad = δ(func) if sym is False else from_triu_1d(δ(func))
            grad = grad.reshape(*shape, *trax)
            trax = (1,) if len(trax) == 0 else trax
            return hess, grad, f(func).reshape(*trax)
        else:
            return hess

    return evaluate_hessian


def jacobian(fun, wrt=0, ntrax=0, parallel=False, full_output=False):
    "Evaluate a scalar-valued function."

    @wraps(fun)
    def evaluate_jacobian(*args, **kwargs):
        args, kwargs, shape, trax = add_tensor(
            args, kwargs, wrt, ntrax, False, True, False
        )
        func = fun(*args, **kwargs)

        if full_output:
            return δ(func), f(func).reshape(*func.shape, *trax)
        else:
            return δ(func)

    return evaluate_jacobian


def gradient_vector_product(fun, wrt=0, ntrax=0, parallel=False):
    "Evaluate the gradient-vector-product of a function."

    @wraps(fun)
    def evaluate_gradient_vector_product(*args, δx, **kwargs):
        args, kwargs, shape, trax = add_tensor(
            args, kwargs, wrt, ntrax, False, gradient=True, δx=δx
        )
        return δ(fun(*args, **kwargs)).reshape(*trax)

    return evaluate_gradient_vector_product


def hessian_vector_product(fun, wrt=0, ntrax=0, parallel=False):
    "Evaluate the hessian-vector-product of a function."

    @wraps(fun)
    def evaluate_hessian_vector_product(*args, δx, **kwargs):
        args, kwargs, shape, trax = add_tensor(
            args, kwargs, wrt, ntrax, False, hessian=True, δx=δx
        )
        return Δδ(fun(*args, **kwargs)).reshape(*shape, *trax)

    return evaluate_hessian_vector_product


def hessian_vectors_product(fun, wrt=0, ntrax=0, parallel=False):
    "Evaluate the hessian-vectors-product of a function."

    @wraps(fun)
    def evaluate_hessian_vectors_product(*args, δx, Δx, **kwargs):
        args, kwargs, shape, trax = add_tensor(
            args, kwargs, wrt, ntrax, False, hessian=True, δx=δx, Δx=Δx
        )
        return Δδ(fun(*args, **kwargs)).reshape(*trax)

    return evaluate_hessian_vectors_product
