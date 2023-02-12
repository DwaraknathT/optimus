#include <pybind11/pybind11.h>
#include "optimus/kernels/ops/gemm.h"

namespace py = pybind11;
namespace opt = optimus;

void invokeGeMM(int a, int b) {
    opt::ops::pybind_test(a, b);
}

PYBIND11_MODULE(pyoptimus, m) {
    m.doc() = "pybind11 example plugin"; // optional module docstring
    m.def("matmul", &invokeGeMM, "A function that adds two numbers");
}
