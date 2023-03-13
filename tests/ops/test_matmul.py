from pyoptimus import math 
import numpy as np 

a = np.random.rand(128, 128).astype(np.float32)
b = np.random.rand(128, 128).astype(np.float32)
c = np.zeros((128, 128)).astype(np.float32)
reference = np.matmul(a, b) 

math.matmul(a, b, c) 
print(np.sum(np.isclose(c, reference, atol=1e-4)) / c.size)

assert np.allclose(c, reference, atol=1e-4)
