cmake_minimum_required(VERSION 3.10)

add_library(log_utils SHARED log_utils.h)
set_property(TARGET log_utils PROPERTY POSITION_INDEPENDENT_CODE  ON)
set_property(TARGET log_utils PROPERTY LINKER_LANGUAGE CXX)

add_library(memanager SHARED memanager.cc)
set_property(TARGET memanager PROPERTY POSITION_INDEPENDENT_CODE  ON)
set_property(TARGET memanager PROPERTY CUDA_RESOLVE_DEVICE_SYMBOLS  ON)
set_property(TARGET memanager PROPERTY CUDA_SEPARABLE_COMPILATION ON)

add_library(cuda_utils SHARED cuda_utils.cu)
set_property(TARGET cuda_utils PROPERTY POSITION_INDEPENDENT_CODE  ON)
set_property(TARGET cuda_utils PROPERTY CUDA_RESOLVE_DEVICE_SYMBOLS  ON)
set_property(TARGET cuda_utils PROPERTY CUDA_SEPARABLE_COMPILATION ON)