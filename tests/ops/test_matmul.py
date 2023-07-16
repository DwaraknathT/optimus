import time
import statistics
import numpy as np 
from pyoptimus import math 

m = 32*2048
n = 1024
k = 1024*3
a = np.random.rand(m, n).astype(np.float32)
b = np.random.rand(n, k).astype(np.float32)

n_tests = 10  # number of tests to perform
execution_times = []

def get_exection_time(func, a, b):
    for i in range(n_tests):
        start_time = time.time()
        output = func(a, b) 
        end_time = time.time()
        execution_time = end_time - start_time
        execution_times.append(execution_time)

    mean_time = statistics.mean(execution_times)
    stdev_time = statistics.stdev(execution_times)

    return output, mean_time, stdev_time

ref_output, ref_mean_time, ref_stdev_time = get_exection_time(np.matmul, a, b)
print('Reference Mean execution time:', ref_mean_time)
print('Reference Standard deviation:', ref_stdev_time)

opt_output, opt_mean_time, opt_stdev_time = get_exection_time(math.matmul, a, b)
print('Optimus Mean execution time:', opt_mean_time)
print('Optimus Standard deviation:', opt_stdev_time)

print(np.sum(np.isclose(opt_output, ref_output, atol=1e-4)) / opt_output.size)

print(ref_output)
print(opt_output)

assert np.allclose(opt_output, ref_output, atol=1e-4)
