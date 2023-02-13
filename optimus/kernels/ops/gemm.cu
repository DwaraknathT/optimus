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
A = (M x N), B = (N x K). Then result C = (M x K). 

c[i][j] = sum (A[i][k] * B[k][i] for k in range (0 to N))
In naive implementation, each thread operates on one element 
of C. Each block can have a max of 1024 threads. 

*/
template<typename T> 
__global__ void naiveGeMMKernel(T* A, T* B, T* C, 
                                const uint32_t M, 
                                const uint32_t N, 
                                const uint32_t K, 
                                const float alpha, 
                                const float beta) {
    // Each thread operates on a single element in the result matrix. 
    // id = no of blocks * dim of blocks + thread idx in block. 
    const uint32_t row = blockIdx.x * blockDim.x + threadIdx.x; 
    const uint32_t col = blockIdx.y * blockDim.y + threadIdx.y; 

    if (row < M && col < K) {
        T dot_product = 0;
        for (size_t i = 0; i < N; i++) {
            dot_product += A[row * N + i] * B[i * K + col];
        }
        C[row * K + col] = dot_product;
    }

}

/* General matrix multiplication kernel. 
    C = A * B 
To check if we can multiply A and B, the number of rows 
of A must match the number of columns of B. 
*/
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

    dim3 gridDim(div_ceil(M, 32), div_ceil(K, 32), 1);
    dim3 blockDim(32, 32, 1); 
    // Launch the kernel 
    naiveGeMMKernel<T><<<gridDim, blockDim>>>(A, B, C, M, N, K, alpha, beta);
}

template void InvokeGeMM<int>(int* A,
                              int* B, 
                              int* C, 
                              const uint32_t M, 
                              const uint32_t N, 
                              const uint32_t K, 
                              const float alpha, 
                              const float beta);

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