add_executable(tensor_test tensor.cu)
target_link_libraries(tensor_test PUBLIC memanager gemm cuda_utils)