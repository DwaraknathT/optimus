#include <iostream>
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <cuda_runtime.h>

#include "optimus/utils/memanager.h"
#include "optimus/kernels/ops/arithmetic.h"
#include "optimus/utils/log_utils.h" 

namespace py = pybind11;
namespace opt = optimus;

template<typename T> 
void invokeMatMulwrapper(py::array_t<T> a, py::array_t<T> b) {
    // Basic assertions for matmul. 
    auto a_n_dim = a.ndim();
    auto b_n_dim = b.ndim();
    const int M = a.shape(0);
    const int N = a.shape(a_n_dim-1);
    const int K = b.shape(b_n_dim-1);
    std::cout << ("\n %d x %d \n", M, K);
    opt::OPT_CHECK((N == b.shape(0)), 
                    "Outer most dim of tensor A must match inner most dim of tensor B");

    // Move the arrays to CPU pinned memory.
    T *d_a, *d_b, *d_c; 
    cudaMallocHost(&d_a, a.nbytes());
    cudaMallocHost(&d_b, b.nbytes());
    // Result array is of shape M x K if A = M x N, B = N x K. 
    const size_t result_size = sizeof(T) * M * K
    cudaMallocHost(&d_c, result_size);

    // Call the matmul kernel.
    opt::ops::InvokeGeMM(d_a, d_b, d_c, M, N, K);
    
    // Clean up memory.
    cudaFreeHost(d_a);
    cudaFreeHost(d_b);
    cudaFreeHost(d_c);
}

PYBIND11_MODULE(pyoptimus, m) {
    auto math_bindings = m.def_submodule("math", "Ops kernels bindings.");
    math_bindings.def("matmul", &invokeMatMulwrapper<int>, "A function that multiplies two matrices.");
    math_bindings.def("matmul", &invokeMatMulwrapper<float>, "A function that multiplies two matrices.");
}
