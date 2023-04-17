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

import numpy as np
from joblib import Parallel, cpu_count, delayed

from ._tensor import Tensor, Δδ, broadcast_to, f, δ
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
        x = triu_1d(x)

    tensor = Tensor(x=x, ntrax=ntrax)
    trax = tensor.trax

    tensor.init(gradient=gradient, hessian=hessian, sym=sym, δx=δx, Δx=Δx)

    if sym:
        tensor = from_triu_1d(tensor)

    args_new[wrt] = tensor

    return args_out, kwargs_out, tensor.shape, trax


def partition(args, kwargs, wrt, ntrax, parallel, chunks=None, batch=100, axis=None):
    """Partition function (keyword) arguments into a list of (keyword) arguments. Only
    top-level args and kwargs with equal shapes to be splitted are allowed."""

    # deactivate parallel evaluation if no trailing axes are present
    if ntrax == 0:
        parallel = False

    # get shape of trailing axes, define axis and chunks
    # if size of chosen axis is below batch, deactivate parallel evaluation
    if parallel:
        # get shape of trailing axes
        trax = (kwargs[wrt] if isinstance(wrt, str) else args[wrt]).shape[-ntrax:]

        # select axis
        if axis is None:
            axis = -(1 + np.argmax(trax[::-1]))

        # define chunks
        if chunks is None:
            if (trax[axis] // batch) > 0:
                chunks = min(trax[axis] // batch, cpu_count())
            else:
                parallel = False

    if not parallel:
        list_of_args_kwargs = [(args, kwargs)]
        chunks = 1
        axis = -1

    else:
        # generate list with args and kwargs for chunks
        list_of_args_kwargs = [[list(args), {**kwargs}] for chunk in range(chunks)]

        # test if object has attribute shape (is tensor or array)
        def isactive(x):
            return hasattr(x, "shape") and np.all(np.isin(trax, x.shape[-ntrax:]))

        # iterate through args and split tensor-like objects
        args_partitioned = []
        for i, arg in enumerate(args):
            if isactive(arg):
                args_partitioned.append((i, np.array_split(arg, chunks, axis=axis)))

        # replace arguments by chunks
        for i, arg in enumerate(list_of_args_kwargs):
            for j, argp in args_partitioned:
                list_of_args_kwargs[i][0][j] = argp[i]

        # iterate through kwargs and split tensor-like objects
        kwargs_partitioned = []
        for key, value in kwargs.items():
            if isactive(value):
                kwargs_partitioned.append(
                    (key, np.array_split(value, chunks, axis=axis))
                )

        # replace keyword arguments by chunks
        for i, kwarg in enumerate(list_of_args_kwargs):
            for key, value in kwargs_partitioned:
                list_of_args_kwargs[i][1][key] = value[i]

    return list_of_args_kwargs, chunks, axis


def concatenate_results(res, axis, full_output):
    "Concatenate results (with optional full output) on existing axis."

    def concat(arrays, axis):
        "Concatenate arrays, fall-back to first item if shape of first array is zero."

        if len(arrays[0].shape) == 0 or len(arrays) == 1:
            return arrays[0]
        else:
            return np.concatenate(arrays, axis=axis)

    if full_output:
        nres = len(res[0])
        return [concat([r[a] for r in res], axis=axis) for a in range(nres)]
    else:
        return concat(res, axis=axis)


def function(fun, wrt=0, ntrax=0, parallel=False):
    "Evaluate a scalar-valued function."

    @wraps(fun)
    def evaluate_function(*args, **kwargs):
        def kernel(args, kwargs):
            args, kwargs, shape, trax = add_tensor(
                args, kwargs, wrt, ntrax, False, False, False
            )
            func = fun(*args, **kwargs)
            return f(func)

        list_of_args_kwargs, chunks, axis = partition(
            args, kwargs, wrt, ntrax, parallel
        )

        res = Parallel(n_jobs=chunks, prefer="threads")(
            delayed(kernel)(*args_chunk) for args_chunk in list_of_args_kwargs
        )

        return concatenate_results(res=res, axis=axis, full_output=False)

    return evaluate_function


def gradient(fun, wrt=0, ntrax=0, parallel=False, full_output=False, sym=False):
    "Evaluate a scalar-valued function."

    @wraps(fun)
    def evaluate_gradient(*args, **kwargs):
        def kernel(args, kwargs):
            args, kwargs, shape, trax = add_tensor(
                args, kwargs, wrt, ntrax, sym, True, False
            )
            func = fun(*args, **kwargs)
            grad = δ(func) if sym is False else from_triu_1d(δ(func))
            grad = broadcast_to(grad, (*shape, *trax))
            if full_output:
                trax = (1,) if len(trax) == 0 else trax
                zeros = np.zeros_like(shape) if sym is False else (0,)
                funct = f(func)[(*zeros,)]
                funct = broadcast_to(funct, (*trax,))
                return grad, funct
            else:
                return grad

        list_of_args, chunks, axis = partition(args, kwargs, wrt, ntrax, parallel)

        res = Parallel(n_jobs=chunks, prefer="threads")(
            delayed(kernel)(*args_chunk) for args_chunk in list_of_args
        )

        return concatenate_results(res=res, axis=axis, full_output=full_output)

    return evaluate_gradient


def hessian(fun, wrt=0, ntrax=0, parallel=False, full_output=False, sym=False):
    "Evaluate a scalar-valued function."

    @wraps(fun)
    def evaluate_hessian(*args, **kwargs):
        def kernel(args, kwargs):
            args, kwargs, shape, trax = add_tensor(
                args, kwargs, wrt, ntrax, sym, False, True
            )
            func = fun(*args, **kwargs)
            hess = Δδ(func) if sym is False else from_triu_2d(Δδ(func))

            if full_output:
                grad = δ(func) if sym is False else from_triu_1d(δ(func))
                zeros = np.zeros_like(shape) if sym is False else (0,)
                grad = grad[(*[slice(None) for a in shape], *zeros)]
                grad = broadcast_to(grad, (*shape, *trax))
                funct = f(func)[
                    (
                        *zeros,
                        *zeros,
                    )
                ]
                funct = broadcast_to(funct, (*trax,))
                trax = (1,) if len(trax) == 0 else trax
                return hess, grad, funct
            else:
                return hess

        list_of_args, chunks, axis = partition(args, kwargs, wrt, ntrax, parallel)

        res = Parallel(n_jobs=chunks, prefer="threads")(
            delayed(kernel)(*args_chunk) for args_chunk in list_of_args
        )

        return concatenate_results(res=res, axis=axis, full_output=full_output)

    return evaluate_hessian


def jacobian(fun, wrt=0, ntrax=0, parallel=False, full_output=False):
    "Evaluate a scalar-valued function."

    @wraps(fun)
    def evaluate_jacobian(*args, **kwargs):
        def kernel(args, kwargs):
            args, kwargs, shape, trax = add_tensor(
                args, kwargs, wrt, ntrax, False, True, False
            )
            func = fun(*args, **kwargs)

            if full_output:
                return δ(func), f(func).reshape(*func.shape, *trax)
            else:
                return δ(func)

        list_of_args, chunks, axis = partition(args, kwargs, wrt, ntrax, parallel)

        res = Parallel(n_jobs=chunks, prefer="threads")(
            delayed(kernel)(*args_chunk) for args_chunk in list_of_args
        )

        return concatenate_results(res=res, axis=axis, full_output=full_output)

    return evaluate_jacobian


def gradient_vector_product(fun, wrt=0, ntrax=0, parallel=False):
    "Evaluate the gradient-vector-product of a function."

    @wraps(fun)
    def evaluate_gradient_vector_product(*args, δx, **kwargs):
        args, kwargs, shape, trax = add_tensor(
            args, kwargs, wrt, ntrax, False, gradient=True, δx=δx
        )
        δfun = δ(fun(*args, **kwargs))
        trim = np.zeros(len(δfun.shape) - ntrax, dtype=int)
        return broadcast_to(δfun[(*trim,)], trax)

    return evaluate_gradient_vector_product


def hessian_vector_product(fun, wrt=0, ntrax=0, parallel=False):
    "Evaluate the hessian-vector-product of a function."

    @wraps(fun)
    def evaluate_hessian_vector_product(*args, δx, **kwargs):
        args, kwargs, shape, trax = add_tensor(
            args, kwargs, wrt, ntrax, False, hessian=True, δx=δx
        )
        Δδfun = Δδ(fun(*args, **kwargs))
        trim = np.zeros(len(Δδfun.shape) - len(shape) - ntrax, dtype=int)
        return broadcast_to(Δδfun[(*trim,)], (*shape, *trax))

    return evaluate_hessian_vector_product


def hessian_vectors_product(fun, wrt=0, ntrax=0, parallel=False):
    "Evaluate the hessian-vectors-product of a function."

    @wraps(fun)
    def evaluate_hessian_vectors_product(*args, δx, Δx, **kwargs):
        args, kwargs, shape, trax = add_tensor(
            args, kwargs, wrt, ntrax, False, hessian=True, δx=δx, Δx=Δx
        )
        Δδfun = Δδ(fun(*args, **kwargs))
        trim = np.zeros(len(Δδfun.shape) - ntrax, dtype=int)
        return broadcast_to(Δδfun[(*trim,)], trax)

    return evaluate_hessian_vectors_product
