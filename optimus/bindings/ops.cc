/*
Python bindings for all the math ops kernels defined in ops namespace.
*/ 
#include <sys/time.h>
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>

#include "optimus/kernels/ops/gemm.h"

namespace opt = optimus; 

void pybind_test_wrapper(int a, int b) {
    opt::ops::pybind_test(a, b);
}

PYBIND11_MODULE(optimus, m) {
    m.def("add", &pybind_test_wrapper);
}
