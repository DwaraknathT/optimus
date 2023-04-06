#include <cmath>
#include <iostream>
#include "optimus/kernels/ops/affine_transform.h"
#include "optimus/kernels/ops/gemm_utils.cuh"
#include "optimus/utils/array_utils.h"
// #include "optimus/utils/cuda_utils.h"

namespace optimus {
namespace ops {

#define WARP_SIZE 32

template <typename T, const int M_chunk_size, const int N_chunk_size,
          const int K_chunk_size, const int M_warp_tile_size,
          const int K_warp_tile_size, const int M_warp_subtile_iters,
          const int K_warp_subtile_iters, const int M_warp_subtile_size,
          const int K_warp_subtile_size, const int result_tile_rows,
          const int result_tile_cols, const int NUM_THREADS>
__global__ void __launch_bounds__(NUM_THREADS)
    AffineTransformKernel(const T *__restrict__ A, const T *__restrict__ B,
                          const T *__restrict__ bias, T *C, const uint32_t M,
                          const uint32_t N, const uint32_t K, bool use_relu) {

    __shared__ T A_chunk[N_chunk_size][M_chunk_size + 1];
    __shared__ T B_chunk[N_chunk_size][K_chunk_size + 1];

    const int warp_id = threadIdx.x / WARP_SIZE;
    const int warp_row = warp_id / (K_chunk_size / K_warp_tile_size);
    const int warp_col = warp_id % (K_chunk_size / K_warp_tile_size);

    const int thread_id_in_warp = threadIdx.x % WARP_SIZE;
    const int warp_thread_cols =
        (K_warp_tile_size / K_warp_subtile_iters) / result_tile_cols;
    const int warp_thread_rows = WARP_SIZE / warp_thread_cols;
    const int thread_row_in_warp = thread_id_in_warp / warp_thread_cols;
    const int thread_col_in_warp = thread_id_in_warp % warp_thread_cols;

    register T results_per_thread[M_warp_subtile_iters][K_warp_subtile_iters]
                                 [result_tile_rows][result_tile_cols] = {0};

    for (int chunk_idx = 0; chunk_idx < (((N - 1) / N_chunk_size) + 1);
         chunk_idx++) {

        if (M % 4 == 0 && N % 4 == 0 && K % 4 == 0) {
            Float4VectorizedSMeMLoad<T, M_chunk_size, N_chunk_size,
                                     K_chunk_size>(A, B, A_chunk, B_chunk,
                                                   chunk_idx, M, N, K);
        } else {
            NonVectorizedSMeMLoad<T, M_chunk_size, N_chunk_size, K_chunk_size>(
                A, B, A_chunk, B_chunk, chunk_idx, M, N, K);
        }

        __syncthreads();

#pragma unroll
        for (int inner_dim = 0; inner_dim < N_chunk_size; inner_dim++) {

#pragma unroll
            for (int M_warp_subtile_iter = 0;
                 M_warp_subtile_iter < M_warp_subtile_iters;
                 M_warp_subtile_iter++) {
#pragma unroll
                for (int K_warp_subtile_iter = 0;
                     K_warp_subtile_iter < K_warp_subtile_iters;
                     K_warp_subtile_iter++) {
#pragma unroll
                    for (int result_row_offset = 0;
                         result_row_offset < result_tile_rows;
                         result_row_offset++) {
#pragma unroll
                        for (int result_col_offset = 0;
                             result_col_offset < result_tile_cols;
                             result_col_offset++) {
                            // Go to the warp tile
                            int current_result_row_in_C =
                                warp_row * M_warp_tile_size;
                            // Go to the warp sub tile
                            current_result_row_in_C +=
                                M_warp_subtile_iter * M_warp_subtile_size;
                            // Go to result row
                            current_result_row_in_C +=
                                result_row_offset * warp_thread_rows +
                                thread_row_in_warp;

                            // Go to the warp tile
                            int current_result_col_in_C =
                                warp_col * K_warp_tile_size;
                            // Go to the warp sub tile
                            current_result_col_in_C +=
                                K_warp_subtile_iter * K_warp_subtile_size;
                            // Go to result col
                            current_result_col_in_C +=
                                result_col_offset * warp_thread_cols +
                                thread_col_in_warp;

                            results_per_thread[M_warp_subtile_iter]
                                              [K_warp_subtile_iter]
                                              [result_row_offset]
                                              [result_col_offset] +=
                                A_chunk[inner_dim][current_result_row_in_C] *
                                B_chunk[inner_dim][current_result_col_in_C];
                        }
                    }
                }
            }
        }

        __syncthreads();
    }

#pragma unroll
    for (int M_warp_subtile_iter = 0;
         M_warp_subtile_iter < M_warp_subtile_iters; M_warp_subtile_iter++) {
#pragma unroll
        for (int K_warp_subtile_iter = 0;
             K_warp_subtile_iter < K_warp_subtile_iters;
             K_warp_subtile_iter++) {
#pragma unroll
            for (int result_row_offset = 0;
                 result_row_offset < result_tile_rows; result_row_offset++) {
#pragma unroll
                for (int result_col_offset = 0;
                     result_col_offset < result_tile_cols;
                     result_col_offset++) {

                    // Go to the warp tile
                    int current_result_row_in_C = warp_row * M_warp_tile_size;
                    // Go to the warp sub tile
                    current_result_row_in_C +=
                        M_warp_subtile_iter * M_warp_subtile_size;
                    // Go to result row
                    current_result_row_in_C +=
                        result_row_offset * warp_thread_rows +
                        thread_row_in_warp;
                    const int current_result_row =
                        (blockIdx.x * M_chunk_size) + current_result_row_in_C;

                    // Go to the warp tile
                    int current_result_col_in_C = warp_col * K_warp_tile_size;
                    // Go to the warp sub tile
                    current_result_col_in_C +=
                        K_warp_subtile_iter * K_warp_subtile_size;
                    // Go to result col
                    current_result_col_in_C +=
                        result_col_offset * warp_thread_cols +
                        thread_col_in_warp;
                    const int current_result_col =
                        (blockIdx.y * K_chunk_size) + current_result_col_in_C;

                    if (current_result_row < M && current_result_col < K) {
                        auto res = results_per_thread[M_warp_subtile_iter]
                                                     [K_warp_subtile_iter]
                                                     [result_row_offset]
                                                     [result_col_offset] +
                                   bias[current_result_col];
                        auto res_after_activation =
                            (use_relu) ? std::max(res, (T)0) : res;
                        C[current_result_row * K + current_result_col] =
                            res_after_activation;
                    }
                }
            }
        }
    }
}

template <typename T>
void InvokeAffineTransformation(T *A, T *B, T *bias, T *C, const uint32_t M,
                                const uint32_t N, const uint32_t K,
                                bool use_relu) {

    const int M_chunk_size = 128;
    const int N_chunk_size = 32;
    const int K_chunk_size = 64;

    const int threads = 128;  // 4 warps.
    // Each warp is operating on a tile of (64, 32)
    const int M_warp_tile_size = 64;
    const int K_warp_tile_size = 32;
    // Each thread computes a result matrix of (4, 4)
    const int result_tile_rows = 4;
    const int result_tile_cols = 4;

    // Divide the warp into subwarps.
    const int K_warp_subtile_iters = 1;
    // 2048 / 512 = 4;
    const int M_warp_subtile_iters = (M_warp_tile_size * K_warp_tile_size) /
                                     (WARP_SIZE * result_tile_rows *
                                      result_tile_cols * K_warp_subtile_iters);
    // 32 threads process a chunk of (8, 64) elements at a time
    // Thread arrangement in warp sub tile = (2, 16)
    const int M_warp_subtile_size =
        M_warp_tile_size / M_warp_subtile_iters;  // 64 / 4 = 8
    const int K_warp_subtile_size =
        K_warp_tile_size / K_warp_subtile_iters;  // 32 / 1 = 32

    dim3 gridDim(div_ceil(M, M_chunk_size), div_ceil(K, K_chunk_size));
    dim3 blockDim(threads);

    // Launch the kernel
    AffineTransformKernel<T, M_chunk_size, N_chunk_size, K_chunk_size,
                          M_warp_tile_size, K_warp_tile_size,
                          M_warp_subtile_iters, K_warp_subtile_iters,
                          M_warp_subtile_size, K_warp_subtile_size,
                          result_tile_rows, result_tile_cols, threads>
        <<<gridDim, blockDim>>>(A, B, bias, C, M, N, K, use_relu);
    // CHECK_LAST_CUDA_ERROR();
}

template void InvokeAffineTransformation<int>(int *A, int *B, int *bias, int *C,
                                              const uint32_t M,
                                              const uint32_t N,
                                              const uint32_t K, bool use_relu);

template void InvokeAffineTransformation<float>(float *A, float *B, float *bias,
                                                float *C, const uint32_t M,
                                                const uint32_t N,
                                                const uint32_t K,
                                                bool use_relu);

}  // namespace ops
}  // namespace optimus