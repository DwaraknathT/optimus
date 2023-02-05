// GEneral Matrix Multiplication Kernel in CUDA C++ 
#include "optimus/kernels/ops/gemm.h"

namespace optimus {
    // A nested namespace for all the core math ops 
    namespace ops {
        // Cuda Kernel to add two N-D tensors of same shape. 
        // c[i] = a[i] + b[i] 
        template<typename T> 
        __global__ void GeMMKernel(T* a, T* b, T* result) {
            *result = *a + *b;
        }

        /* General matrix multiplication kernel. 
            C = A * B 
        To check if we can multiply A and B, the number of rows 
        of A must match the number of columns of B. 
        */
        int InvokeGeMM(int a, int b) {
            int c;
            int *d_a, *d_b, *d_c; // device copies of a, b, c
            int size = sizeof(int);

            // Allocate space for device copies of a, b, c
            cudaMalloc((void **)&d_a, size);
            cudaMalloc((void **)&d_b, size);
            cudaMalloc((void **)&d_c, size);
            cudaMemcpy(d_a, &a, size, cudaMemcpyHostToDevice);
            cudaMemcpy(d_b, &b, size, cudaMemcpyHostToDevice);
            
            // Launch add() kernel on GPU
            GeMMKernel<<<1,1>>>(d_a, d_b, d_c);
            // Copy result back to host
            cudaMemcpy(&c, d_c, size, cudaMemcpyDeviceToHost);
            // Cleanup
            cudaFree(d_a); cudaFree(d_b); cudaFree(d_c);

            return c;
        }
    }
}