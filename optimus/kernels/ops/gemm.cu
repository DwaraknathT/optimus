#include <iostream>
#include <cmath>
#include "optimus/kernels/ops/gemm.h"
#include "optimus/utils/array_utils.h"

namespace optimus {
// A nested namespace for all the core math ops 
namespace ops {

/*
Simple general matrix multiplication kernel. 
C = alpha * (A * B) + beta * C 
Let's say we are multiplying 2 matrices A and B. 
A = (M x N), B = (N x K). Then result C = (M x K). 

c[i][j] = sum (A[i][k] * B[k][i] for k in range (0 to N))
*/
template<typename T, const int chunk_size> 
__global__ void GeMMKernel(T* A, T* B, T* C, 
                            const uint32_t M, 
                            const uint32_t N, 
                            const uint32_t K, 
                            const float alpha, 
                            const float beta) {
    // Each thread operates on a single element in the result matrix. 
    const int thread_row = threadIdx.x / chunk_size; 
    const int thread_col = threadIdx.x % chunk_size; 

    const uint32_t row = blockIdx.x * chunk_size + thread_row; 
    const uint32_t col = blockIdx.y * chunk_size + thread_col; 

    __shared__ T A_chunk[chunk_size][chunk_size];
    __shared__ T B_chunk[chunk_size][chunk_size];

    T temp_dot_prods[chunk_size] = {0};
    for (int chunk_idx = 0; chunk_idx < (((N - 1) / chunk_size) + 1); chunk_idx++) {

        for (int current_thread_row = 0; current_thread_row < chunk_size; current_thread_row++) {
            const int current_row = blockIdx.x * chunk_size + current_thread_row;

            if (current_row < M && (chunk_idx * chunk_size + thread_col) < N) {
                const int A_index = (current_row * N) + (chunk_idx * chunk_size + thread_col);
                A_chunk[current_thread_row][thread_col] = A[A_index];
            }
            else {
                A_chunk[current_thread_row][thread_col] = 0; 
            }
            
            if ((chunk_idx * chunk_size + current_thread_row) < N && (col < K)) {
                const int B_index = (chunk_idx * chunk_size + current_thread_row) * K + col; 
                B_chunk[current_thread_row][thread_col] = B[B_index];
            }
            else {
                B_chunk[current_thread_row][thread_col] = 0;
            }
        }

        __syncthreads();

        if (row < M && col < K) {
            for (int i = 0; i < chunk_size; i++) {
                T B_cache = B_chunk[i][thread_col];
                for (int j = 0; j < chunk_size; j++) {
                    const int current_row = blockIdx.x * chunk_size + j;
                    temp_dot_prods[j] += A_chunk[j][i] * B_cache;
                }
            }
        }

        __syncthreads();
    } 

    if (row < M && col < K) {
        for (int j = 0; j < chunk_size; j++) {
            const int current_row = blockIdx.x * chunk_size + j;
            C[current_row * K + col] = temp_dot_prods[j];
        }
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
    const int chunk_size = 32; 
    dim3 gridDim(div_ceil(M, chunk_size), div_ceil(K, chunk_size));
    dim3 blockDim(chunk_size); 
    // Launch the kernel 
    GeMMKernel<T, chunk_size><<<gridDim, blockDim>>>(A, B, C, M, N, K, alpha, beta);
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