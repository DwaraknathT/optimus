from pyoptimus import math 
import numpy as np 

a = np.random.rand(2, 3).astype(np.float32)
b = np.random.rand(3, 2).astype(np.float32)
c = np.zeros((2, 2)).astype(np.float32)
reference = np.matmul(a, b) 

math.matmul(a, b, c) 
assert np.array_equal(c, reference)
