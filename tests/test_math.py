import tensortrax as tr
import tensortrax.math as tm
import numpy as np


def test_math():
    
    F = np.ones((3, 3))
    T = tr.Tensor(F)
    
    assert isinstance(F @ T, tr.Tensor)
    assert isinstance(T @ F, tr.Tensor)
    assert isinstance(T * F, tr.Tensor)
    assert isinstance(F * T, tr.Tensor)
    
    assert isinstance(F / T, tr.Tensor)
    assert isinstance(F * T, tr.Tensor)
    
    assert tm.array.cross(F, F).shape == F.shape
    assert tm.array.eye(F).shape == F.shape


if __name__ == "__main__":
    test_math()