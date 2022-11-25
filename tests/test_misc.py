import tensortrax as ttr
from tensortrax import dot, ddot, transpose, det, inv
import numpy as np


def test_tensor():

    np.random.seed(125161)
    d = np.tile(np.eye(3).reshape(3, 3, 1), (1, 1, 8000)).reshape(3, 3, 8, 1000)
    f = d + np.random.rand(3, 3, 8, 1000) / 10

    I = ttr.Constant(d)
    F = ttr.Tensor(f)
    C = dot(transpose(F), F)

    I1 = ddot(F, F)
    J = det(F)

    W = (J ** (-2 / 3) * I1 - 3) / 2
    S = det(F) ** (-2 / 3) * (I - I1 / 3 * inv(C))
    P = dot(F, S).real
    A = dot(F, S).dual

    assert np.allclose(W.dual, P.real)
    assert A.shape == (3, 3, 3, 3, 8, 1000)


if __name__ == "__main__":
    test_tensor()
