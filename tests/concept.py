def fun(x, y):
    z = x * x
    return z

def grad(fun, argnum=0):
    
    def inner(*args, dual=True, **kwargs):
        return fun(*args, **kwargs)
    
    return inner

f = fun(2, 3)
dfdx = grad(fun, argnum=0)(2, 3)

#x = Tensor(array, dual=False)
#y = Tensor(array, dual=True)

class HyperDualNumber:
    
    def __init__(self, f, e1=1, e2=1, e1e2=0):
        self.f = f
        self.e1 = e1
        self.e2 = e2
        self.e1e2 = e1e2
    
    def __repr__(self):
        return f"{self.f} + {self.e1}e1 + {self.e2}e2 + {self.e1e2}e1e2"
    
    def __add__(self, b):
        a = self
        return HyperDualNumber(
            f=a.f+b.f, e1=a.e1+b.e1, e2=a.e2+b.e2, e1e2=a.e1e2+b.e1e2
        )
    
    def __mul__(self, b):
        a = self
        return HyperDualNumber(
            f=a.f*b.f, e1=a.f*b.e1+a.e1*b.f, e2=a.f*b.e2+a.e2*b.f, e1e2=a.f*b.e1e2+a.e1*b.e2+a.e2*b.e1++a.e1e2*b.f
        )
    
x = HyperDualNumber(0.4)