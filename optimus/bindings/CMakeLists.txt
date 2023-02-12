pybind11_add_module(pyoptimus MODULE ops.cc)
target_link_libraries(pyoptimus PUBLIC gemm_kernels)