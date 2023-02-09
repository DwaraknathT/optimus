cmake_minimum_required(VERSION 3.10)

add_library(optimus_ops SHARED ops.cc)
target_include_directories(optimus_ops PUBLIC ${CMAKE_SOURCE_DIR}/extern/pybind11/include)
target_link_libraries(optimus_ops PUBLIC Python3::Python)
