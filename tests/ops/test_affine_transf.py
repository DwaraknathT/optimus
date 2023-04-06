import time
import statistics
import numpy as np 
import jax.numpy as jnp
from pyoptimus import math 

m = 32*2048
n = 1024
k = 1024*3
a = np.random.rand(m, n).astype(np.float32)
b = np.random.rand(n, k).astype(np.float32)
bias = np.random.rand(k).astype(np.float32)

n_tests = 10  # number of tests to perform
execution_times = []

ref_output = jnp.matmul(a, b) + bias 
opt_output = math.affine_transform(a, b, bias)

print(np.sum(np.isclose(opt_output, ref_output, atol=1e-4)) / opt_output.size)

assert np.allclose(opt_output, ref_output, atol=1e-4)

