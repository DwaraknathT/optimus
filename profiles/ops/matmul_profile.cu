#include <cublas_v2.h>
#include <curand.h>
#include <iostream>
#include "optimus/ops/gemm.h"
#include "optimus/tensor.h"

using namespace optimus;

void GPU_fill_rand(float* A, int nr_rows_A, int nr_cols_A) {
    // Create a pseudo-random number generator
    curandGenerator_t prng;
    curandCreateGenerator(&prng, CURAND_RNG_PSEUDO_DEFAULT);
    // Set the seed for the random number generator using the system clock
    curandSetPseudoRandomGeneratorSeed(prng, (unsigned long long)clock());
    // Fill the array with random numbers on the device
    curandGenerateUniform(prng, A, nr_rows_A * nr_cols_A);
}

void gpu_blas_mmul(const float* A, const float* B, float* C, const int m,
                   const int k, const int n) {
    int lda = m, ldb = k, ldc = m;
    const float alf = 1;
    const float bet = 0;
    const float* alpha = &alf;
    const float* beta = &bet;
    // Create a handle for CUBLAS
    cublasHandle_t handle;
    cublasCreate(&handle);
    // Do the actual multiplication
    cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, m, n, k, alpha, A, lda, B,
                ldb, beta, C, ldc);
    // Destroy the handle
    cublasDestroy(handle);
}

void test_matmul() {
    const uint32_t m = 32*2048;
    const uint32_t n = 1024;
    const uint32_t k = 1024*3;
    const float alpha = 1.0;
    const float beta = 0;

    auto A_tensor = new optimus::Tensor<float>({m, n}, optimus::MEMORY_GPU);
    auto B_tensor = new optimus::Tensor<float>({n, k}, optimus::MEMORY_GPU);
    auto C_tensor = new optimus::Tensor<float>({m, k}, optimus::MEMORY_GPU);
        
    GPU_fill_rand(A_tensor->data, m, n);
    GPU_fill_rand(B_tensor->data, n, k);

    optimus::ops::InvokeGeMM<float>(A_tensor, B_tensor, C_tensor, alpha, beta);
    gpu_blas_mmul(A_tensor->data, B_tensor->data, C_tensor->data, m, n, k);
}


int main() {
    test_matmul();
    return 0;
}