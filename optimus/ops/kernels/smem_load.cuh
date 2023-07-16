#pragma once

#include "optimus/tensor.h"

namespace optimus {
namespace ops {
template <typename T, const int M_chunk_size, const int N_chunk_size,
          const int K_chunk_size>
__device__ void NonVectorizedSMeMLoad(const Tensor<T> *__restrict__ A,
                                      const Tensor<T> *__restrict__ B,
                                      T A_chunk[][M_chunk_size + 1],
                                      T B_chunk[][K_chunk_size + 1],
                                      const int chunk_idx, const int M,
                                      const int N, const int K) {
    const int A_thread_cols = N_chunk_size;
    const int A_thread_rows = blockDim.x / A_thread_cols;
    const int thread_row_in_A = threadIdx.x / A_thread_cols;
    const int thread_col_in_A = threadIdx.x % A_thread_cols;

    const int B_thread_cols = K_chunk_size;
    const int B_thread_rows = blockDim.x / B_thread_cols;
    const int thread_row_in_B = threadIdx.x / B_thread_cols;
    const int thread_col_in_B = threadIdx.x % B_thread_cols;

    for (int A_row_offset = 0; A_row_offset < M_chunk_size;
         A_row_offset += A_thread_rows) {
        const int current_thread_row_in_A = thread_row_in_A + A_row_offset;
        const int current_A_row =
            (blockIdx.x * M_chunk_size) + current_thread_row_in_A;
        const int current_A_col = chunk_idx * N_chunk_size + thread_col_in_A;
        if (current_A_row < M && current_A_col < N) {
            A_chunk[thread_col_in_A][current_thread_row_in_A] =
                A->get({current_A_row, current_A_col});
        } else {
            A_chunk[thread_col_in_A][current_thread_row_in_A] = 0;
        }
    }

    for (int B_row_offset = 0; B_row_offset < N_chunk_size;
         B_row_offset += B_thread_rows) {
        const int current_thread_row_in_B = thread_row_in_B + B_row_offset;
        const int current_B_row =
            chunk_idx * N_chunk_size + current_thread_row_in_B;
        const int current_B_col = (blockIdx.y * K_chunk_size) + thread_col_in_B;
        if (current_B_row < N && current_B_col < K) {
            B_chunk[current_thread_row_in_B][thread_col_in_B] =
                B->get({current_B_row, current_B_col});
        } else {
            B_chunk[current_thread_row_in_B][thread_col_in_B] = 0;
        }
    }
}

template <typename T, const int M_chunk_size, const int N_chunk_size,
          const int K_chunk_size>
__device__ void Float4VectorizedSMeMLoad(const Tensor<T> *__restrict__ A,
                                         const Tensor<T> *__restrict__ B,
                                         T A_chunk[][M_chunk_size + 1],
                                         T B_chunk[][K_chunk_size + 1],
                                         const int chunk_idx, const int M,
                                         const int N, const int K) {
    const int A_thread_cols = N_chunk_size / 4;
    const int A_thread_rows = blockDim.x / A_thread_cols;
    const int thread_row_in_A = threadIdx.x / A_thread_cols;
    const int thread_col_in_A = threadIdx.x % A_thread_cols;

    const int B_thread_cols = K_chunk_size / 4;
    const int B_thread_rows = blockDim.x / B_thread_cols;
    const int thread_row_in_B = threadIdx.x / B_thread_cols;
    const int thread_col_in_B = threadIdx.x % B_thread_cols;

    for (int A_row_offset = 0; A_row_offset < M_chunk_size;
         A_row_offset += A_thread_rows) {
        const int current_thread_row_in_A = thread_row_in_A + A_row_offset;
        const int current_A_row =
            (blockIdx.x * M_chunk_size) + current_thread_row_in_A;
        const int current_A_col =
            chunk_idx * N_chunk_size + thread_col_in_A * 4;
        const int A_index = current_A_row * N + current_A_col;

        // const float4 A_load = reinterpret_cast<const float4
        // *>(&A[A_index])[0];
        const float4 A_load = reinterpret_cast<const float4 *>(
            &A->get({current_A_row, current_A_col}))[0];
        A_chunk[thread_col_in_A * 4 + 0][current_thread_row_in_A] = A_load.x;
        A_chunk[thread_col_in_A * 4 + 1][current_thread_row_in_A] = A_load.y;
        A_chunk[thread_col_in_A * 4 + 2][current_thread_row_in_A] = A_load.z;
        A_chunk[thread_col_in_A * 4 + 3][current_thread_row_in_A] = A_load.w;
    }

    for (int B_row_offset = 0; B_row_offset < N_chunk_size;
         B_row_offset += B_thread_rows) {
        const int current_thread_row_in_B = thread_row_in_B + B_row_offset;
        const int current_B_row =
            chunk_idx * N_chunk_size + current_thread_row_in_B;
        const int current_B_col =
            (blockIdx.y * K_chunk_size) + thread_col_in_B * 4;
        const int B_index = current_B_row * K + current_B_col;

        // const float4 B_load = reinterpret_cast<const float4
        // *>(&B[B_index])[0];
        const float4 B_load = reinterpret_cast<const float4 *>(
            &B->get({current_B_row, current_B_col}))[0];
        B_chunk[current_thread_row_in_B][thread_col_in_B * 4 + 0] = B_load.x;
        B_chunk[current_thread_row_in_B][thread_col_in_B * 4 + 1] = B_load.y;
        B_chunk[current_thread_row_in_B][thread_col_in_B * 4 + 2] = B_load.z;
        B_chunk[current_thread_row_in_B][thread_col_in_B * 4 + 3] = B_load.w;
    }
}

}  // namespace ops
}  // namespace optimus