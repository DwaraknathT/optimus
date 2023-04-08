#include <cublas_v2.h>
#include <curand.h>
#include <iostream>
#include "optimus/kernels/ops/gemm.h"
#include "optimus/kernels/ops/affine_transform.h"
#include "optimus/utils/memanager.h"
#include "optimus/layers/dense.h"
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
    const uint32_t m = 32 * 2048;
    const uint32_t n = 1024;
    const uint32_t k = 1024 * 3;
    const size_t size_a = sizeof(float) * m * n;
    const size_t size_b = sizeof(float) * n * k;
    const size_t size_c = sizeof(float) * m * k;
    const float alpha = 1.0;
    const float beta = 0;

    auto memory_manager = new optimus::MemManager();
    float* h_c =
        (float*)(memory_manager->allocate(size_c, optimus::MEMORY_CPU));

    float* d_a =
        (float*)(memory_manager->allocate(size_a, optimus::MEMORY_GPU));
    float* d_b =
        (float*)(memory_manager->allocate(size_b, optimus::MEMORY_GPU));
    float* d_c =
        (float*)(memory_manager->allocate(size_c, optimus::MEMORY_GPU));
        
    GPU_fill_rand(d_a, m, n);
    GPU_fill_rand(d_b, n, k);

    optimus::ops::InvokeGeMM<float>(d_a, d_b, d_c, m, n, k, alpha, beta);
    gpu_blas_mmul(d_a, d_b, d_c, m, n, k);

    cudaMemcpy(d_c, h_c, size_c, cudaMemcpyDeviceToHost);
    delete memory_manager;
}

void test_affine_transform() {
    const uint32_t m = 32 * 2048;
    const uint32_t n = 1024;
    const uint32_t k = 1024 * 3;
    const size_t size_a = sizeof(float) * m * n;
    const size_t size_b = sizeof(float) * n * k;
    const size_t size_bias = sizeof(float) * k;
    const size_t size_c = sizeof(float) * m * k;

    auto memory_manager = new optimus::MemManager();
    float* h_c =
        (float*)(memory_manager->allocate(size_c, optimus::MEMORY_CPU));

    float* d_a =
        (float*)(memory_manager->allocate(size_a, optimus::MEMORY_GPU));
    float* d_b =
        (float*)(memory_manager->allocate(size_b, optimus::MEMORY_GPU));
    float* d_bias =
        (float*)(memory_manager->allocate(size_bias, optimus::MEMORY_GPU));
    float* d_c =
        (float*)(memory_manager->allocate(size_c, optimus::MEMORY_GPU));

    GPU_fill_rand(d_a, m, n);
    GPU_fill_rand(d_b, n, k);
    GPU_fill_rand(d_bias, 1, k);

    optimus::ops::InvokeAffineTransformation<float>(d_a, d_b, d_bias, d_c, m, n, k);

    cudaMemcpy(d_c, h_c, size_c, cudaMemcpyDeviceToHost);
    delete memory_manager;
}

int main() {
    auto a = optimus::Tensor<int>({32, 1024, 512});
    auto stride = a.stride();
    for (int i : stride) {
        std::cout << i << " ";
    }
    // auto layer = new optimus::layers::Dense<float>(32, 64, optimus::MEMORY_GPU);
    // delete layer;
    // test_affine_transform();
    return 0;
}
