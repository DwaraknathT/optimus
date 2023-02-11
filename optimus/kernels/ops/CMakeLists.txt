cmake_minimum_required(VERSION 3.10)

add_library(gemm_kernels SHARED gemm.cu)
set_property(TARGET gemm_kernels PROPERTY POSITION_INDEPENDENT_CODE  ON)
set_property(TARGET gemm_kernels PROPERTY CUDA_RESOLVE_DEVICE_SYMBOLS  ON)
