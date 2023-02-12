cmake_minimum_required(VERSION 3.10)

add_library(memanager SHARED memanager.cc)
set_property(TARGET memanager PROPERTY POSITION_INDEPENDENT_CODE  ON)
set_property(TARGET memanager PROPERTY CUDA_RESOLVE_DEVICE_SYMBOLS  ON)
set_property(TARGET memanager PROPERTY CUDA_SEPARABLE_COMPILATION ON)