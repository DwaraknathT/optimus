#include <iostream>
#include "optimus/kernels/ops/gemm.h"
#include "optimus/utils/memanager.h"

using namespace optimus;

int main() {
    const uint32_t m = 4096;
    const uint32_t n = 4096;
    const uint32_t k = 4096; 
    const size_t size_a = sizeof(float) * m * n;
    const size_t size_b = sizeof(float) * n * k; 
    const size_t size_c = sizeof(float) * m * k; 
    const float alpha = 1.0;
    const float beta = 0; 

    auto memory_manager = new optimus::MemManager();
    float* h_a = (float*)(memory_manager->allocate(size_a, optimus::MEMORY_CPU)); 
    float* h_b = (float*)(memory_manager->allocate(size_b, optimus::MEMORY_CPU)); 
    float* h_c = (float*)(memory_manager->allocate(size_c, optimus::MEMORY_CPU));

    // random initialize matrix A
    for (uint32_t i = 0; i < m; ++i) {
        for (uint32_t j = 0; j < n; ++j) {
            h_a[i * n + j] = (float)((i * n + j) % 1024);
        }
    }

    // random initialize matrix B
    for (uint32_t i = 0; i < n; ++i) {
        for (uint32_t j = 0; j < k; ++j) {
            h_b[i * k + j] = (float)((i * n + j) % 1024);
        }
    }

    float* d_a = (float*)(memory_manager->allocate(size_a, optimus::MEMORY_GPU)); 
    float* d_b = (float*)(memory_manager->allocate(size_b, optimus::MEMORY_GPU)); 
    float* d_c = (float*)(memory_manager->allocate(size_c, optimus::MEMORY_GPU));

    cudaMemcpy(d_a, h_a, size_a, cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, h_b, size_b, cudaMemcpyHostToDevice);
    

    optimus::ops::InvokeGeMM<float>(d_a, d_b, d_c, m, n, k, alpha, beta);
    cudaMemcpy(d_c, h_c, size_c, cudaMemcpyDeviceToHost);
    delete memory_manager;

    return 0;
}
