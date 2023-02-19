from pyoptimus import math 
import numpy as np 

a = np.random.rand(48, 128).astype(np.float32)
b = np.random.rand(128, 64).astype(np.float32)
c = np.zeros((48, 64)).astype(np.float32)
reference = np.matmul(a, b) 
# print("=============")
# print(a)
# print("=============")
# print(b)

math.matmul(a, b, c) 

print("=============")
print(c)
print("=============")
print(reference)

print(np.sum(np.isclose(c, reference, atol=1e-4)) / c.size)

assert np.allclose(c, reference, atol=1e-4)
