if '__file__' in globals():
    import os, sys
    sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
import numpy as np
from dezero.core import Variable
import dezero.functions as F

x = Variable(np.array([1.0, 2.0]))
y = Variable(np.array([4.0, 5.0]))

def f(x):
    t = x ** 2
    y = F.sum(t)
    return y

y = f(x)
y.backward(create_graph=True)

gx = x.grad
x.cleargrad()
z = F.matmul(v, gx)  # (1)
z.backward()  # (2)
print(x.grad)