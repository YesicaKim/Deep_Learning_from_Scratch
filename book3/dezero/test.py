import numpy as np
from dezero import Variable

def f(x):
    y = x ** 4 - 2 * x ** 2
    return y

x = Variable(np.array(2.0))
y = f(x)
y.backward(create_graph=True)  # (1)
print(x.grad)

# 두 번째 역전파 진행
gx = x.grad   # (2)
gx.backward() # (3)
print(x.grad)