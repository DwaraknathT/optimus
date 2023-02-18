from pyoptimus import math 
import numpy as np 

a = np.random.rand(7, 12).astype(np.float32)
b = np.random.rand(12, 13).astype(np.float32)
c = np.zeros((7, 13)).astype(np.float32)
reference = np.matmul(a, b) 

math.matmul(a, b, c) 
assert np.array_equal(c, reference)
