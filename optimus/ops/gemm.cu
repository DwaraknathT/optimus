#include <cmath>
#include <iostream>
#include "optimus/ops/gemm.h"
#include "optimus/ops/kernels/fp32_gemm.cuh"
#include "optimus/tensor.h"
#include "optimus/utils/array_utils.h"
#include "optimus/utils/cuda_utils.h"

namespace optimus {
// A nested namespace for all the core math ops
namespace ops {

#define WARP_SIZE 32

template <typename T>
void InvokeGeMM(Tensor<T> *A, Tensor<T> *B, Tensor<T> *C, const float alpha,
                const float beta) {

    const int M = A->shape_[0];
    const int N = A->shape_[1];
    const int K = B->shape_[1];

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
    FP32_GeMM_Kernel<T, M_chunk_size, N_chunk_size, K_chunk_size,
                     M_warp_tile_size, K_warp_tile_size, M_warp_subtile_iters,
                     K_warp_subtile_iters, M_warp_subtile_size,
                     K_warp_subtile_size, result_tile_rows, result_tile_cols,
                     threads><<<gridDim, blockDim>>>(A->data, B->data, C->data,
                                                     M, N, K, alpha, beta);
    CHECK_LAST_CUDA_ERROR();
}

template void InvokeGeMM<int>(Tensor<int> *A, Tensor<int> *B, Tensor<int> *C,
                              const float alpha, const float beta);

template void InvokeGeMM<float>(Tensor<float> *A, Tensor<float> *B,
                                Tensor<float> *C, const float alpha,
                                const float beta);

}  // namespace ops
}  // namespace optimus