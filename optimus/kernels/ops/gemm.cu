#include <iostream>
#include <cmath>
#include "optimus/kernels/ops/gemm.h"
#include "optimus/utils/array_utils.h"

namespace optimus {
// A nested namespace for all the core math ops 
namespace ops {

template<typename T, const int chunk_size, const int results_per_thread> 
__global__ void GeMMKernel(T* A, T* B, T* C, 
                            const uint32_t M, 
                            const uint32_t N, 
                            const uint32_t K, 
                            const float alpha, 
                            const float beta) {
    // Each thread operates on a single element in the result matrix. 
    const int thread_row = threadIdx.x / blockDim.x; 
    const int thread_col = threadIdx.x % blockDim.x; 

    const uint32_t row = blockIdx.x * blockDim.x + thread_row; 
    const uint32_t col = blockIdx.y * blockDim.x + thread_col; 

    __shared__ T A_chunk[chunk_size][chunk_size];
    __shared__ T B_chunk[chunk_size][chunk_size];

    T dot_products[results_per_thread] = {0};
    
    for (int chunk_idx = 0; chunk_idx < (((N - 1) / chunk_size) + 1); chunk_idx++) {
        for (int current_result = 0; current_result < results_per_thread; current_result++) {
            const int current_row = row + current_result;
            const int current_t_row = thread_row + current_result; 

            if (current_row < M && (chunk_idx * chunk_size + thread_col) < N) {
                const int A_index = (current_row * N) + (chunk_idx * chunk_size + thread_col);
                A_chunk[current_result][thread_col] = A[A_index];
            }
            else {
                A_chunk[current_result][thread_col] = 0; 
            }
            
            if ((chunk_idx * chunk_size + current_t_row) < N && (col < K)) {
                const int B_index = (chunk_idx * chunk_size + current_t_row) * K + col; 
                B_chunk[current_result][thread_col] = B[B_index];
            }
            else {
                B_chunk[current_result][thread_col] = 0;
            }
        }

        __syncthreads();

        for (int i = 0; i < chunk_size; i++) {
            T B_cache = B_chunk[i][thread_col];
            for (int current_result = 0; current_result < results_per_thread; current_result++) {
                dot_products[current_result] += A_chunk[current_result][i] * B_cache;
            }
        }

        __syncthreads();
    } 

    for (int current_result = 0; current_result < results_per_thread; current_result++) {
        const int current_row = row + current_result;
        if (current_row < M && col < K) {
            C[current_row * K + col] = dot_products[current_result];
        }
    }
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
    const int chunk_size = 32; 
    const int threads = 32;
    const int results_pre_thread = 32;
    dim3 gridDim(div_ceil(M, chunk_size), div_ceil(K, chunk_size));
    dim3 blockDim(threads); 
    // Launch the kernel 
    GeMMKernel<T, chunk_size, results_pre_thread><<<gridDim, blockDim>>>(A, B, C, M, N, K, alpha, beta);
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