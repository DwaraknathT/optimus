from pyoptimus import math 
import numpy as np 

m = 123
n = 567 
k = 328
a = np.random.rand(m, n).astype(np.float32)
b = np.random.rand(n, k).astype(np.float32)
c = np.zeros((m, k)).astype(np.float32)
reference = np.matmul(a, b) 

math.matmul(a, b, c) 
print(np.sum(np.isclose(c, reference, atol=1e-4)) / c.size)

assert np.allclose(c, reference, atol=1e-4)
