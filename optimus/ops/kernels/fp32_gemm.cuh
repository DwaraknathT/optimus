#include <cooperative_groups.h>
#include <cmath>
#include <iostream>
#include "optimus/ops/kernels/smem_load.cuh"
#include "optimus/utils/array_utils.h"
#include "optimus/utils/cuda_utils.h"

namespace optimus {
namespace ops {

namespace cg = cooperative_groups;

#define WARP_SIZE 32

template <typename T, const int M_chunk_size, const int N_chunk_size,
          const int K_chunk_size, const int M_warp_tile_size,
          const int K_warp_tile_size, const int M_warp_subtile_iters,
          const int K_warp_subtile_iters, const int M_warp_subtile_size,
          const int K_warp_subtile_size, const int result_tile_rows,
          const int result_tile_cols, const int NUM_THREADS>
__global__ void __launch_bounds__(NUM_THREADS)
    FP32_GeMM(const T *__restrict__ A, const T *__restrict__ B, T *C,
               const uint32_t M, const uint32_t N, const uint32_t K,
               const float alpha, const float beta) {

    __shared__ T A_chunk[N_chunk_size][M_chunk_size + 1];
    __shared__ T B_chunk[N_chunk_size][K_chunk_size + 1];

    register T results_per_thread[M_warp_subtile_iters][K_warp_subtile_iters]
                                 [result_tile_rows][result_tile_cols] = {0};

    cg::thread_block thread_block = cg::this_thread_block();

    const int warp_id = threadIdx.x / WARP_SIZE;
    const int warp_row = warp_id / (K_chunk_size / K_warp_tile_size);
    const int warp_col = warp_id % (K_chunk_size / K_warp_tile_size);

    const int thread_id_in_warp = threadIdx.x % WARP_SIZE;
    const int warp_thread_cols =
        (K_warp_tile_size / K_warp_subtile_iters) / result_tile_cols;
    const int warp_thread_rows = WARP_SIZE / warp_thread_cols;
    const int thread_row_in_warp = thread_id_in_warp / warp_thread_cols;
    const int thread_col_in_warp = thread_id_in_warp % warp_thread_cols;

    for (int chunk_idx = 0; chunk_idx < (((N - 1) / N_chunk_size) + 1);
         chunk_idx++) {

        // if (M % 4 == 0 && N % 4 == 0 && K % 4 == 0) {
        //     Float4VectorizedSMeMLoad<T, M_chunk_size, N_chunk_size,
        //                              K_chunk_size>(A, B, A_chunk, B_chunk,
        //                                            chunk_idx, M, N, K);
        // } else {
        NonVectorizedSMeMLoad<T, M_chunk_size, N_chunk_size, K_chunk_size>(
            A, B, A_chunk, B_chunk, chunk_idx, M, N, K);
        // }

        thread_block.sync();

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

        thread_block.sync();
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
                        auto result = results_per_thread[M_warp_subtile_iter]
                                                        [K_warp_subtile_iter]
                                                        [result_row_offset]
                                                        [result_col_offset];
                        C->set({current_result_row, current_result_col},
                               result);
                    }
                }
            }
        }
    }
}

}  // namespace ops
}  // namespace optimus