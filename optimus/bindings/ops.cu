#include <cuda_runtime.h>
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <iostream>
#include "optimus/kernels/ops/affine_transform.h"
#include "optimus/kernels/ops/gemm.h"
#include "optimus/utils/log_utils.h"
#include "optimus/utils/memanager.h"

namespace py = pybind11;
namespace opt = optimus;

template <typename T>
py::array_t<T> invokeMatMulwrapper(py::array_t<T> a, py::array_t<T> b) {
    auto a_n_dim = a.ndim();
    auto b_n_dim = b.ndim();
    const int M = a.shape(0);
    const int N = a.shape(a_n_dim - 1);
    const int K = b.shape(b_n_dim - 1);
    const float alpha = 1.0;
    const float beta = 0.0;

    // Basic assertions for matmul.
    opt::OPT_CHECK(
        (N == b.shape(0)),
        "Outer most dim of tensor A must match inner most dim of tensor B");

    // Move the arrays to CPU pinned memory.
    py::array_t<T> result(M * K);

    T *d_a, *d_b, *d_result;
    cudaMallocHost(&d_a, a.nbytes());
    cudaMallocHost(&d_b, b.nbytes());
    // Result array is of shape M x K if A = M x N, B = N x K.
    cudaMallocHost(&d_result, result.nbytes());

    cudaMemcpy((void *)d_a, (void *)a.data(), a.nbytes(),
               cudaMemcpyHostToDevice);
    cudaMemcpy((void *)d_b, (void *)b.data(), b.nbytes(),
               cudaMemcpyHostToDevice);

    // Call the matmul kernel.
    opt::ops::InvokeGeMM<T>(d_a, d_b, d_result, M, N, K, alpha, beta);

    cudaMemcpy((void *)result.data(), (void *)d_result, result.nbytes(),
               cudaMemcpyDeviceToHost);
    // Clean up memory.
    cudaFreeHost(d_a);
    cudaFreeHost(d_b);
    cudaFreeHost(d_result);

    result.resize({M, K});

    return result;
}

template <typename T>
py::array_t<T> invokeAffineTransformwrapper(py::array_t<T> a, py::array_t<T> b,
                                            py::array_t<T> bias) {
    auto a_n_dim = a.ndim();
    auto b_n_dim = b.ndim();
    const int M = a.shape(0);
    const int N = a.shape(a_n_dim - 1);
    const int K = b.shape(b_n_dim - 1);

    // Basic assertions for matmul.
    opt::OPT_CHECK(
        (N == b.shape(0)),
        "Outer most dim of tensor A must match inner most dim of tensor B");
    opt::OPT_CHECK(
        (bias.shape(bias.ndim() - 1) == K),
        "Outer most dim of tensor weight must match dimension of bias");

    // Move the arrays to CPU pinned memory.
    py::array_t<T> result(M * K);

    T *d_a, *d_b, *d_bias, *d_result;
    cudaMallocHost(&d_a, a.nbytes());
    cudaMallocHost(&d_b, b.nbytes());
    cudaMallocHost(&d_bias, bias.nbytes());
    // Result array is of shape M x K if A = M x N, B = N x K.
    cudaMallocHost(&d_result, result.nbytes());

    cudaMemcpy((void *)d_a, (void *)a.data(), a.nbytes(),
               cudaMemcpyHostToDevice);
    cudaMemcpy((void *)d_b, (void *)b.data(), b.nbytes(),
               cudaMemcpyHostToDevice);
    cudaMemcpy((void *)d_bias, (void *)bias.data(), bias.nbytes(),
               cudaMemcpyHostToDevice);

    // Call the matmul kernel.
    opt::ops::InvokeAffineTransformation<T>(d_a, d_b, d_bias, d_result, M, N,
                                            K);

    cudaMemcpy((void *)result.data(), (void *)d_result, result.nbytes(),
               cudaMemcpyDeviceToHost);
    // Clean up memory.
    cudaFreeHost(d_a);
    cudaFreeHost(d_b);
    cudaFreeHost(d_bias);
    cudaFreeHost(d_result);

    result.resize({M, K});

    return result;
}

PYBIND11_MODULE(pyoptimus, m) {
    auto math_bindings = m.def_submodule("math", "Ops kernels bindings.");
    math_bindings.def("matmul", &invokeMatMulwrapper<int>,
                      "A function that multiplies two matrices.");
    math_bindings.def("matmul", &invokeMatMulwrapper<float>,
                      "A function that multiplies two matrices.");

    math_bindings.def(
        "affine_transform", &invokeAffineTransformwrapper<int>,
        "A function that multiplies two matrices and adds a bias.");
    math_bindings.def(
        "affine_transform", &invokeAffineTransformwrapper<float>,
        "A function that multiplies two matrices and adds a bias.");
}
