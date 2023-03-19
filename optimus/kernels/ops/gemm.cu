#include <iostream>
#include <cmath>
#include "optimus/kernels/ops/gemm.h"
#include "optimus/utils/array_utils.h"

namespace optimus
{
    // A nested namespace for all the core math ops
    namespace ops
    {

#define WARP_SIZE 32

        template <typename T,
                  const int M_chunk_size,
                  const int N_chunk_size,
                  const int K_chunk_size,
                  const int M_warp_tile_size,
                  const int K_warp_tile_size,
                  const int M_warp_subtile_iters,
                  const int K_warp_subtile_iters,
                  const int M_warp_subtile_size,
                  const int K_warp_subtile_size,
                  const int result_tile_rows,
                  const int result_tile_cols,
                  const int NUM_THREADS>
        __global__ void __launch_bounds__(NUM_THREADS) GeMMKernel(const T *__restrict__ A,
                                                                  const T *__restrict__ B,
                                                                  T *C,
                                                                  const uint32_t M,
                                                                  const uint32_t N,
                                                                  const uint32_t K,
                                                                  const float alpha,
                                                                  const float beta)
        {

            __shared__ T A_chunk[N_chunk_size][M_chunk_size + 1];
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

            const int warp_id = threadIdx.x / WARP_SIZE;
            const int warp_row = warp_id / (K_chunk_size / K_warp_tile_size);
            const int warp_col = warp_id % (K_chunk_size / K_warp_tile_size);

            const int thread_id_in_warp = threadIdx.x % WARP_SIZE;
            const int warp_thread_cols = (K_warp_tile_size / K_warp_subtile_iters) / result_tile_cols;
            const int warp_thread_rows = WARP_SIZE / warp_thread_cols;
            const int thread_row_in_warp = thread_id_in_warp / warp_thread_cols;
            const int thread_col_in_warp = thread_id_in_warp % warp_thread_cols;

            register T A_cache[M_warp_subtile_iters][result_tile_rows] = {0};
            register T B_cache[K_warp_subtile_iters][result_tile_cols] = {0};
            register T results_per_thread[M_warp_subtile_iters][K_warp_subtile_iters][result_tile_rows][result_tile_cols] = {0};

            for (int chunk_idx = 0; chunk_idx < (((N - 1) / N_chunk_size) + 1); chunk_idx++)
            {

                // We want the threads to load A and B in a memory coalesced fashion.
                for (int A_row_offset = 0; A_row_offset < M_chunk_size; A_row_offset += A_thread_rows)
                {

                    const int current_thread_row_in_A = thread_row_in_A + A_row_offset;
                    const int current_A_row = block_row + current_thread_row_in_A;
                    const int current_A_col = chunk_idx * N_chunk_size + thread_col_in_A;
                    const int A_index = current_A_row * N + current_A_col;
                    if (current_A_row < M && current_A_col < N)
                    {
                        A_chunk[thread_col_in_A][current_thread_row_in_A] = A[A_index];
                    }
                    else
                    {
                        A_chunk[thread_col_in_A][current_thread_row_in_A] = 0;
                    }
                }

                // Similarly load B
                for (int B_row_offset = 0; B_row_offset < N_chunk_size; B_row_offset += B_thread_rows)
                {
                    const int current_thread_row_in_B = thread_row_in_B + B_row_offset;
                    const int current_B_row = chunk_idx * N_chunk_size + current_thread_row_in_B;
                    const int current_B_col = block_col + thread_col_in_B;
                    const int B_index = current_B_row * K + current_B_col;
                    if (current_B_row < N && current_B_col < K)
                    {
                        B_chunk[current_thread_row_in_B][thread_col_in_B] = B[B_index];
                    }
                    else
                    {
                        B_chunk[current_thread_row_in_B][thread_col_in_B] = 0;
                    }
                }

                __syncthreads();

#pragma unroll
                for (int inner_dim = 0; inner_dim < N_chunk_size; inner_dim++)
                {
#pragma unroll
                    for (int M_warp_subtile_iter = 0; M_warp_subtile_iter < M_warp_subtile_iters; M_warp_subtile_iter++)
                    {
#pragma unroll
                        for (int result_row_offset = 0; result_row_offset < result_tile_rows; result_row_offset++)
                        {
                            // Go to the warp tile
                            int current_result_row_in_C = warp_row * M_warp_tile_size;
                            // Go to the warp sub tile
                            current_result_row_in_C += M_warp_subtile_iter * M_warp_subtile_size;
                            // Go to result row
                            current_result_row_in_C += result_row_offset * warp_thread_rows + thread_row_in_warp;

                            A_cache[M_warp_subtile_iter][result_row_offset] = A_chunk[inner_dim][current_result_row_in_C];
                        }
                    }

#pragma unroll
                    for (int K_warp_subtile_iter = 0; K_warp_subtile_iter < K_warp_subtile_iters; K_warp_subtile_iter++)
                    {
#pragma unroll
                        for (int result_col_offset = 0; result_col_offset < result_tile_cols; result_col_offset++)
                        {
                            // Go to the warp tile
                            int current_result_col_in_C = warp_col * K_warp_tile_size;
                            // Go to the warp sub tile
                            current_result_col_in_C += K_warp_subtile_iter * K_warp_subtile_size;
                            // Go to result col
                            current_result_col_in_C += result_col_offset * warp_thread_cols + thread_col_in_warp;

                            B_cache[K_warp_subtile_iter][result_col_offset] = B_chunk[inner_dim][current_result_col_in_C];
                        }
                    }

#pragma unroll
                    for (int M_warp_subtile_iter = 0; M_warp_subtile_iter < M_warp_subtile_iters; M_warp_subtile_iter++)
                    {
#pragma unroll
                        for (int K_warp_subtile_iter = 0; K_warp_subtile_iter < K_warp_subtile_iters; K_warp_subtile_iter++)
                        {
#pragma unroll
                            for (int result_row_offset = 0; result_row_offset < result_tile_rows; result_row_offset++)
                            {
#pragma unroll
                                for (int result_col_offset = 0; result_col_offset < result_tile_cols; result_col_offset++)
                                {
                                    results_per_thread[M_warp_subtile_iter][K_warp_subtile_iter][result_row_offset][result_col_offset] +=
                                        A_cache[M_warp_subtile_iter][result_row_offset] * B_cache[K_warp_subtile_iter][result_col_offset];
                                }
                            }
                        }
                    }
                }

                __syncthreads();
            }

#pragma unroll
            for (int M_warp_subtile_iter = 0; M_warp_subtile_iter < M_warp_subtile_iters; M_warp_subtile_iter++)
            {
#pragma unroll
                for (int K_warp_subtile_iter = 0; K_warp_subtile_iter < K_warp_subtile_iters; K_warp_subtile_iter++)
                {
#pragma unroll
                    for (int result_row_offset = 0; result_row_offset < result_tile_rows; result_row_offset++)
                    {
#pragma unroll
                        for (int result_col_offset = 0; result_col_offset < result_tile_cols; result_col_offset++)
                        {

                            // Go to the warp tile
                            int current_result_row_in_C = warp_row * M_warp_tile_size;
                            // Go to the warp sub tile
                            current_result_row_in_C += M_warp_subtile_iter * M_warp_subtile_size;
                            // Go to result row
                            current_result_row_in_C += result_row_offset * warp_thread_rows + thread_row_in_warp;
                            const int current_result_row = block_row + current_result_row_in_C;

                            // Go to the warp tile
                            int current_result_col_in_C = warp_col * K_warp_tile_size;
                            // Go to the warp sub tile
                            current_result_col_in_C += K_warp_subtile_iter * K_warp_subtile_size;
                            // Go to result col
                            current_result_col_in_C += result_col_offset * warp_thread_cols + thread_col_in_warp;
                            const int current_result_col = block_col + current_result_col_in_C;

                            if (current_result_row < M && current_result_col < K)
                            {
                                C[current_result_row * K + current_result_col] = results_per_thread[M_warp_subtile_iter][K_warp_subtile_iter][result_row_offset][result_col_offset];
                            }
                        }
                    }
                }
            }
        }

        template <typename T>
        void InvokeGeMM(T *A,
                        T *B,
                        T *C,
                        const uint32_t M,
                        const uint32_t N,
                        const uint32_t K,
                        const float alpha,
                        const float beta)
        {

            const int M_chunk_size = 64;
            const int N_chunk_size = 32;
            const int K_chunk_size = 128;

            const int threads = 128; // 4 warps.
            // Each warp is operating on a tile of (32, 64)
            const int M_warp_tile_size = 32;
            const int K_warp_tile_size = 64;
            // Each thread computes a result matrix of (4, 4)
            const int result_tile_rows = 4;
            const int result_tile_cols = 4;

            // Divide the warp into subwarps.
            const int K_warp_subtile_iters = 1;
            // 2048 / 512 = 4;
            const int M_warp_subtile_iters = (M_warp_tile_size * K_warp_tile_size) / (WARP_SIZE * result_tile_rows * result_tile_cols * K_warp_subtile_iters);
            // 32 threads process a chunk of (8, 64) elements at a time
            // Thread arrangement in warp sub tile = (2, 16)
            const int M_warp_subtile_size = M_warp_tile_size / M_warp_subtile_iters; // 32 / 4 = 8
            const int K_warp_subtile_size = K_warp_tile_size / K_warp_subtile_iters; // 64 / 1 = 64

            dim3 gridDim(div_ceil(M, M_chunk_size), div_ceil(K, K_chunk_size));
            dim3 blockDim(threads);

            printf("Launching grid with dims x: %d, y: %d \n", gridDim.x, gridDim.y);
            printf("Threads in a block %d \n", blockDim.x);

            // Launch the kernel
            GeMMKernel<T,
                       M_chunk_size,
                       N_chunk_size,
                       K_chunk_size,
                       M_warp_tile_size,
                       K_warp_tile_size,
                       M_warp_subtile_iters,
                       K_warp_subtile_iters,
                       M_warp_subtile_size,
                       K_warp_subtile_size,
                       result_tile_rows,
                       result_tile_cols,
                       threads><<<gridDim, blockDim>>>(A, B, C, M, N, K, alpha, beta);
        }

        template void InvokeGeMM<int>(int *A,
                                      int *B,
                                      int *C,
                                      const uint32_t M,
                                      const uint32_t N,
                                      const uint32_t K,
                                      const float alpha,
                                      const float beta);

        template void InvokeGeMM<float>(float *A,
                                        float *B,
                                        float *C,
                                        const uint32_t M,
                                        const uint32_t N,
                                        const uint32_t K,
                                        const float alpha,
                                        const float beta);

    } // namespace ops
} // namespace optimus