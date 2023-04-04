cmake_minimum_required(VERSION 3.10)

add_library(gemm_kernels SHARED gemm.cu)
set_property(TARGET gemm_kernels PROPERTY POSITION_INDEPENDENT_CODE  ON)
set_property(TARGET gemm_kernels PROPERTY CUDA_RESOLVE_DEVICE_SYMBOLS  ON)
set_property(TARGET gemm_kernels PROPERTY CUDA_SEPARABLE_COMPILATION ON)

add_library(affine_transforms SHARED affine_transform.cu)
set_property(TARGET affine_transforms PROPERTY POSITION_INDEPENDENT_CODE  ON)
set_property(TARGET affine_transforms PROPERTY CUDA_RESOLVE_DEVICE_SYMBOLS  ON)
set_property(TARGET affine_transforms PROPERTY CUDA_SEPARABLE_COMPILATION ON)

add_library(arithmetic_kernels SHARED arithmetic.cu)
set_property(TARGET arithmetic_kernels PROPERTY POSITION_INDEPENDENT_CODE  ON)
set_property(TARGET arithmetic_kernels PROPERTY CUDA_RESOLVE_DEVICE_SYMBOLS  ON)
set_property(TARGET arithmetic_kernels PROPERTY CUDA_SEPARABLE_COMPILATION ON)