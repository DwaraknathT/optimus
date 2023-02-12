#include <iostream>
#include "optimus/kernels/ops/gemm.h"
#include "optimus/utils/memanager.h"

using namespace optimus;

int main() {
    const uint32_t m = 70;
    const uint32_t n = 128;
    const uint32_t k = 256; 
    const size_t size_a = sizeof(float) * m * n;
    const size_t size_b = sizeof(float) * n * k; 
    const size_t size_c = sizeof(float) * m * k; 
    const float alpha = 1.0;
    const float beta = 0; 

    auto memory_manager = new optimus::MemManager();
    float* h_a = (float*)(memory_manager->allocate(size_a, optimus::MEMORY_CPU_PINNED)); 
    float* h_b = (float*)(memory_manager->allocate(size_b, optimus::MEMORY_CPU_PINNED)); 
    float* h_c = (float*)(memory_manager->allocate(size_c, optimus::MEMORY_CPU_PINNED));

    // random initialize matrix A
    for (int i = 0; i < m; ++i) {
        for (int j = 0; j < n; ++j) {
            h_a[i * n + j] = (float)((i * n + j) % 1024);
        }
    }

    // random initialize matrix B
    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < k; ++j) {
            h_b[i * k + j] = (float)((i * n + j) % 1024);
        }
    }

    delete memory_manager;
    // optimus::ops::InvokeGeMM<float>(h_a, h_b, h_c, m, n, k, alpha, beta);

    return 0;
}
