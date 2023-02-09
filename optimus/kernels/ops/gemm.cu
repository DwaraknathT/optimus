// General Matrix Multiplication Kernel in CUDA C++ 
#include <iostream>
#include <cmath>
#include "optimus/kernels/ops/gemm.h"

namespace optimus {
// A nested namespace for all the core math ops 
namespace ops {

/*
Simple general matrix multiplication kernel. 
C = alpha * (A * B) + beta * C 
Let's say we are multiplying 2 matrices A and B. 
A - M x N, B - N x K. Result C - M x K 
c[i][j] = sum (A[i][k] * B[k][i] for k in range (0 to N))
In naive implementation, each thread operates on one element 
of C. Each block can have a max of 1024 threads. 
no of blocks = (M/32, K/32)
No of threads = (32, 32)
threadIdx.x = row id 
threadIdx.y = col id 
global x = blockIdx.x * blockDim.x + threadIdx.x 
global y = blockIdx.y 
*/
// template<typename T> 
// __global__ void GeMMKernel(T* a, T* b, T* result) {
// }

/* General matrix multiplication kernel. 
    C = A * B 
To check if we can multiply A and B, the number of rows 
of A must match the number of columns of B. 
*/
void pybind_test(int a, int b) {
    printf("Inside pybind");
}

template <typename T>
void InvokeGeMM(T* A,
                T* B, 
                T* C, 
                const uint32_t M, 
                const uint32_t N, 
                const uint32_t K, 
                const float alpha, 
                const float beta) {
    // Naive Implementation - each thread in the kernels operates on one 
    // element in the result matrix. Each block can have 1024 threads, so 
    // threadDim is (32, 32) and Block dim is (M / 32, k / 32).

    std::cout << (float)(M / 32) << std::endl;
    return;
}

template void InvokeGeMM<float>(float* A,
                                float* B, 
                                float* C, 
                                const uint32_t M, 
                                const uint32_t N, 
                                const uint32_t K, 
                                const float alpha, 
                                const float beta);

} // namespace ops 
} // namespace optimus