cmake_minimum_required(VERSION 3.15)

add_library(gemm STATIC gemm.cu)
set_target_properties(gemm PROPERTY CUDA_RESOLVE_DEVICE_SYMBOLS  ON)
