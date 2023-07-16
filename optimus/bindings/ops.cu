#include <cuda_runtime.h>
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <iostream>
#include "optimus/ops/gemm.h"
#include "optimus/utils/log_utils.h"
#include "optimus/utils/memanager.h"

namespace py = pybind11;
namespace opt = optimus;

template <typename T>
py::array_t<T> invokeMatMulwrapper(py::array_t<T> A, py::array_t<T> B) {
    auto A_n_dim = A.ndim();
    auto B_n_dim = B.ndim();
    const int M = A.shape(0);
    const int N = A.shape(A_n_dim - 1);
    const int K = B.shape(B_n_dim - 1);
    const float alpha = 1.0;
    const float beta = 0.0;

    // Basic assertions for matmul.
    opt::OPT_CHECK(
        (N == B.shape(0)),
        "Outer most dim of tensor A must match inner most dim of tensor B");

    // Create tensor objects
    std::vector<int> A_shape =
        std::vector<int>(A.shape(), A.shape() + A.ndim());
    auto A_tensor = new optimus::Tensor<T>(A_shape, optimus::MEMORY_GPU);
    A_tensor->set(A.data());

    std::vector<int> B_shape =
        std::vector<int>(B.shape(), B.shape() + B.ndim());
    auto B_tensor = new optimus::Tensor<T>(B_shape, optimus::MEMORY_GPU);
    B_tensor->set(B.data());

    auto result_tensor = new optimus::Tensor<T>({M, K}, optimus::MEMORY_GPU);

    // Call the matmul kernel.
    opt::ops::InvokeGeMM<T>(A_tensor, B_tensor, result_tensor, alpha, beta);

    // Move the arrays to CPU pinned memory.
    py::array_t<T> result(M * K);
    cudaMemcpy((void *)result.data(), (void *)result_tensor->data,
               result.nbytes(), cudaMemcpyDeviceToHost);

    result.resize({M, K});

    return result;
}


PYBIND11_MODULE(pyoptimus, m) {
    auto math_bindings = m.def_submodule("math", "Ops kernels bindings.");
    math_bindings.def("matmul", &invokeMatMulwrapper<int>,
                      "A function that multiplies two matrices.");
    math_bindings.def("matmul", &invokeMatMulwrapper<float>,
                      "A function that multiplies two matrices.");
}
