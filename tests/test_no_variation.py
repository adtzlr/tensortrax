import numpy as np

import tensortrax as tr
import tensortrax.math as tm


def deformation(seed=6, npoints=4):
    "Return a random right Cauchy-Green deformation tensor with a trailing axis."
    np.random.seed(seed)
    F = np.eye(3).reshape(3, 3, 1) + (np.random.rand(3, 3, npoints) - 0.5) / 5
    return np.einsum("ki...,kj...->ij...", F, F)


def internal(C):
    "An internal (state) variable, which depends on the deformation."
    Ci = tm.base.eye(C) + C @ C / 2
    return tm.linalg.det(Ci) ** (-1 / 3) * Ci


def psi(C, Ci):
    "A function of the deformation and of the internal variable."
    return tm.trace(C @ tm.linalg.inv(Ci)) + tm.trace(C @ C @ Ci) / 4


def fun(C):
    return psi(C, tm.no_variation(internal(C)))


def detach(A):
    "Return a Tensor which is constant in the variation and in the linearization."
    zero = tr.ZERO
    return tr.Tensor(tr.f(A), δx=zero, Δx=zero, Δδx=zero, ntrax=A.ntrax, ndual=A.ndual)


def fun_fixed(C):
    return psi(C, detach(internal(C)))


def fun_total(C):
    return psi(C, internal(C))


def test_no_variation_tensor():
    C = tr.Tensor(deformation())
    C.init(hessian=True)
    Ci = internal(C)

    for A in [tm.no_variation(Ci), Ci.no_variation()]:
        assert np.allclose(tr.f(A), tr.f(Ci))
        assert np.allclose(tr.Δ(A), tr.Δ(Ci))
        assert tr.δ(A) is tr.ZERO
        assert tr.Δδ(A) is tr.ZERO
        assert A.ntrax == Ci.ntrax
        assert A.ndual == Ci.ndual

    # an array has no variation and is returned unchanged
    x = deformation()
    assert tm.no_variation(x) is x


def test_no_variation_gradient():
    C = deformation()

    # the gradient is the partial derivative at fixed internal variables
    dψdC = tr.gradient(fun_fixed, ntrax=1)(C)
    assert np.allclose(tr.gradient(fun, ntrax=1)(C), dψdC)

    # the internal variable contributes to the gradient without `no_variation()`
    dWdC = tr.gradient(fun_total, ntrax=1)(C)
    assert not np.allclose(dWdC, dψdC)


def test_no_variation_hessian():
    C = deformation()

    # the hessian is the consistent linearization of the gradient
    d2WdCdC = tr.hessian(fun, ntrax=1)(C)
    d2WdCdC_fd = np.zeros_like(d2WdCdC)

    h = 1e-6
    for i in range(3):
        for j in range(3):
            dC = np.zeros_like(C)
            dC[i, j] = h
            dWdC_plus = tr.gradient(fun, ntrax=1)(C + dC)
            dWdC_minus = tr.gradient(fun, ntrax=1)(C - dC)
            d2WdCdC_fd[:, :, i, j] = (dWdC_plus - dWdC_minus) / (2 * h)

    assert np.allclose(d2WdCdC, d2WdCdC_fd, rtol=1e-6, atol=1e-6)

    # the linearization of the internal variable is required for the hessian
    d2ψdCdC = tr.hessian(fun_fixed, ntrax=1)(C)
    assert not np.allclose(d2WdCdC, d2ψdCdC)


def test_no_variation_real_to_dual():
    C = tr.Tensor(deformation())
    C.init(hessian=True)
    A = tm.linalg.inv(internal(C))
    x = C @ C

    for mul in [None, tm.special.ddot]:
        W1 = tm.real_to_dual(A, x, mul=mul)
        W2 = (lambda A, B: A * B if mul is None else mul(A, B))(tm.no_variation(A), x)

        assert np.allclose(tr.δ(W1), tr.δ(W2))
        assert np.allclose(tr.Δδ(W1), tr.Δδ(W2))


if __name__ == "__main__":
    test_no_variation_tensor()
    test_no_variation_gradient()
    test_no_variation_hessian()
    test_no_variation_real_to_dual()
