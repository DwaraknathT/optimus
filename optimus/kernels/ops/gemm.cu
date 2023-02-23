#include <iostream>
#include <cmath>
#include "optimus/kernels/ops/gemm.h"
#include "optimus/utils/array_utils.h"

namespace optimus {
// A nested namespace for all the core math ops 
namespace ops {

template<typename T, 
         const int M_chunk_size, 
         const int N_chunk_size, 
         const int K_chunk_size, 
         const int result_tile_rows, 
         const int result_tile_cols> 
__global__ void GeMMKernel(T* A, T* B, T* C, 
                            const uint32_t M, 
                            const uint32_t N, 
                            const uint32_t K, 
                            const float alpha, 
                            const float beta) {

    __shared__ T A_chunk[M_chunk_size][N_chunk_size + 1];
    __shared__ T B_chunk[N_chunk_size][K_chunk_size + 1];
    
    const int block_row = blockIdx.x * M_chunk_size;
    const int block_col = blockIdx.y * K_chunk_size; 
 
    const int A_thread_cols = N_chunk_size;
    const int A_thread_rows = blockDim.x / A_thread_cols; 
    const int thread_row_in_A = threadIdx.x / A_thread_cols;
    const int thread_col_in_A = threadIdx.x % A_thread_cols; 

    const int B_thread_cols = K_chunk_size;
    const int B_thread_rows = blockDim.x / B_thread_cols; 
    const int thread_row_in_B = threadIdx.x / B_thread_cols;
    const int thread_col_in_B = threadIdx.x % B_thread_cols; 

    const int C_thread_cols = (K_chunk_size / result_tile_cols);
    const int C_thread_rows = blockDim.x / C_thread_cols;
    const int thread_row_in_C = threadIdx.x / C_thread_cols;
    const int thread_col_in_C = threadIdx.x % C_thread_cols;

    T A_cache[result_tile_rows] = {0};
    T B_cache[result_tile_cols] = {0};

    T results_per_thread[result_tile_rows][result_tile_cols] = {0};

    for (int chunk_idx = 0; chunk_idx < (((N - 1) / N_chunk_size) + 1); chunk_idx++) {

        // We want the threads to load A and B in a memory coalesced fashion. 
        for (int A_row_offset = 0; A_row_offset < M_chunk_size; A_row_offset += A_thread_rows) {

            const int current_thread_row_in_A = thread_row_in_A + A_row_offset;
            const int current_A_row = block_row + current_thread_row_in_A; 
            const int current_A_col = chunk_idx * N_chunk_size + thread_col_in_A; 

            if (current_A_row < M && current_A_col < N) {
                const int A_index = current_A_row * N + current_A_col;
                A_chunk[current_thread_row_in_A][thread_col_in_A] = A[A_index];
            }
            else {
                A_chunk[current_thread_row_in_A][thread_col_in_A] = 0; 
            }
        }

        // Similarly load B 
        for (int B_row_offset = 0; B_row_offset < N_chunk_size; B_row_offset += B_thread_rows) {
            const int current_thread_row_in_B = thread_row_in_B + B_row_offset;
            const int current_B_row = chunk_idx * N_chunk_size + current_thread_row_in_B; 
            const int current_B_col = block_col + thread_col_in_B; 

            if (current_B_row < N && current_B_col < K) {
                const int B_index = current_B_row * K + current_B_col;
                B_chunk[current_thread_row_in_B][thread_col_in_B] = B[B_index];
            }
            else {
                B_chunk[current_thread_row_in_B][thread_col_in_B] = 0; 
            }
        }

        __syncthreads();

        for (int inner_dim = 0; inner_dim < N_chunk_size; inner_dim++) {

            for (int result_row_offset = 0; result_row_offset < result_tile_rows; result_row_offset++) {
                A_cache[result_row_offset] = A_chunk[thread_row_in_C * result_tile_rows + result_row_offset][inner_dim];
            }
            for (int result_col_offset = 0; result_col_offset < result_tile_cols; result_col_offset++) {
                B_cache[result_col_offset] = B_chunk[inner_dim][thread_col_in_C * result_tile_cols + result_col_offset];
            }

            for (int result_row_offset = 0; result_row_offset < result_tile_rows; result_row_offset++) {
                for (int result_col_offset = 0; result_col_offset < result_tile_cols; result_col_offset++) {
                    results_per_thread[result_row_offset][result_col_offset] += A_cache[result_row_offset] * B_cache[result_col_offset];
                }
            }
        }
        __syncthreads();
    }

    for (int result_row_offset = 0; result_row_offset < result_tile_rows; result_row_offset++) {
        for (int result_col_offset = 0; result_col_offset < result_tile_cols; result_col_offset++) {
            const int current_result_row = block_row + thread_row_in_C * result_tile_rows + result_row_offset;
            const int current_result_col = block_col + thread_col_in_C * result_tile_cols + result_col_offset; 

            if (current_result_row < M && current_result_col < K) {
                C[current_result_row * K + current_result_col] = results_per_thread[result_row_offset][result_col_offset];
            }

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
    
    const int M_chunk_size = 32;
    const int N_chunk_size = 32;
    const int K_chunk_size = 32; 
    const int result_tile_rows = 4;
    const int result_tile_cols = 4; 

    // (32 * 32) / 16 = 64 threads 
    const int threads = div_ceil(M_chunk_size * K_chunk_size, result_tile_rows * result_tile_cols);

    dim3 gridDim(div_ceil(M, M_chunk_size), div_ceil(K, K_chunk_size));
    dim3 blockDim(threads); 

    printf("Launching grid with dims x: %d, y: %d \n", gridDim.x, gridDim.y);
    printf("Threads in a block %d \n", blockDim.x);

    // Launch the kernel 
    GeMMKernel<T, 
               M_chunk_size, 
               N_chunk_size, 
               K_chunk_size, 
               result_tile_rows, 
               result_tile_cols><<<gridDim, blockDim>>>(A, B, C, M, N, K, alpha, beta);
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