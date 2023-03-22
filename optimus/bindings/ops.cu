#include <iostream>
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <cuda_runtime.h>

#include "optimus/utils/memanager.h"
#include "optimus/kernels/ops/gemm.h"
#include "optimus/utils/log_utils.h" 

namespace py = pybind11;
namespace opt = optimus;

template<typename T> 
void invokeMatMulwrapper(py::array_t<T> a, py::array_t<T> b, py::array_t<T> c) {
    auto a_n_dim = a.ndim();
    auto b_n_dim = b.ndim();
    const int M = a.shape(0);
    const int N = a.shape(a_n_dim-1);
    const int K = b.shape(b_n_dim-1);
    const float alpha = 1.0; 
    const float beta = 0.0;

    // Basic assertions for matmul. 
    opt::OPT_CHECK((N == b.shape(0)), 
                    "Outer most dim of tensor A must match inner most dim of tensor B");

    // Move the arrays to CPU pinned memory.
    T *d_a, *d_b, *d_c; 
    cudaMallocHost(&d_a, a.nbytes());
    cudaMallocHost(&d_b, b.nbytes());
    // Result array is of shape M x K if A = M x N, B = N x K. 
    cudaMallocHost(&d_c, c.nbytes());

    cudaMemcpy((void*)d_a, (void*)a.data(), a.nbytes(), cudaMemcpyHostToDevice);
    cudaMemcpy((void*)d_b, (void*)b.data(), b.nbytes(), cudaMemcpyHostToDevice);

    // Call the matmul kernel.
    opt::ops::InvokeGeMM<T>(d_a, d_b, d_c, M, N, K, alpha, beta); 
    
    cudaMemcpy((void*)c.data(), (void*)d_c, c.nbytes(), cudaMemcpyDeviceToHost);
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
