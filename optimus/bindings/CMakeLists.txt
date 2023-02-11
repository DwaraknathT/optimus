if(BUILD_PYBIND)
    pybind11_add_module(cmake_example ops.cc)
    target_link_libraries(cmake_example PUBLIC gemm_kernels)
endif()