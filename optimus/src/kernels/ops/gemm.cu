// GEneral Matrix Multiplication Kernel in CUDA C++ 

#include "src/kernels/ops/gemm.h"

namespace optimus {
    // A nested namespace for all the core math ops 
    namespace ops {

        // Cuda Kernel to add two N-D tensors of same shape. 
        // c[i] = a[i] + b[i] 
        template<typename T> 
        __global__ void element_wise_add_kernel(T* a, T* b, T* c) {

        }

        // Element wise addition of 2 tensors. 
        template<typename T> 
        void invokeGeMM(T* a, T* b, T* c) {

        }
    }
}
