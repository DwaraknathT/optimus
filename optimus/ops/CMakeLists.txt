add_subdirectory(kernels)

add_library(gemm SHARED gemm.cu)
set_property(TARGET gemm PROPERTY POSITION_INDEPENDENT_CODE  ON)
set_property(TARGET gemm PROPERTY CUDA_RESOLVE_DEVICE_SYMBOLS  ON)
set_property(TARGET gemm PROPERTY CUDA_SEPARABLE_COMPILATION ON)
