import tensortrax as tr
import tensortrax.math as tm
import numpy as np


def test_math():
    
    F = np.eye(3) + np.arange(9).reshape(3, 3) / 10
    T = tr.Tensor(F)
    
    assert isinstance(F @ T, tr.Tensor)
    assert isinstance(T @ F, tr.Tensor)
    assert isinstance(T * F, tr.Tensor)
    assert isinstance(F * T, tr.Tensor)
    assert isinstance(F * T, tr.Tensor)
    
    assert np.allclose(tm.linalg.det(F), tm.linalg.det(T).x)
    
    for fun in [tm.sin, tm.cos, tm.tan, tm.tanh]:
        assert np.allclose(fun(F), fun(T).x)
        
    for fun in [tm.linalg.det]:
        assert np.allclose(fun(F), fun(T).x)
    
    assert tm.linalg.eigvalsh(T).shape == (3,)
    
    assert tm.array.cross(F, F).shape == F.shape
    assert tm.array.eye(F).shape == F.shape


if __name__ == "__main__":
    test_math()