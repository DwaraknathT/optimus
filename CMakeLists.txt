cmake_minimum_required(VERSION 3.15 FATAL_ERROR)
message("Using CMake version ${CMAKE_VERSION}")

project(optimus LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 20)
# custom cmake macros location
list(INSERT CMAKE_MODULE_PATH 0 "${CMAKE_CURRENT_SOURCE_DIR}/cmake")
list(INSERT CMAKE_MODULE_PATH 0 "${CMAKE_CURRENT_SOURCE_DIR}/cmake/cuda")

# Prevent from build in source tree
include(preventBuildInSource)
# Init build type: Release, Debug, ...
include(initBuildType)
# useful for debugging cmake
include(print_target_properties)
# always export compile commands database
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

enable_language(CUDA)
if(NOT DEFINED CMAKE_CUDA_STANDARD)
    set(CMAKE_CUDA_STANDARD 20)
    set(CMAKE_CUDA_STANDARD_REQUIRED ON)
endif()

set(Python3_FIND_VIRTUALENV "STANDARD")
find_package(Python3 COMPONENTS Interpreter Development)

# we also need extra stuff to make sure compile flags are correctly
# passed to nvcc / host compiler
include(protect_nvcc_flags)
include(protect_pthread_flag)

add_subdirectory(src)

#-------------------------------------------------------------------#
# Configuration summary
#-------------------------------------------------------------------#
message("//===================================================")
message("  ${PROJECT_NAME} build configuration:")
message("//===================================================")
message("  CUDA compiler ID      : ${CMAKE_CUDA_COMPILER_ID}")
message("  CUDA compiler Version : ${CMAKE_CUDA_COMPILER_VERSION}")
message("  C++ Compiler : ${CMAKE_CXX_COMPILER_ID} "
  "${CMAKE_CXX_COMPILER_VERSION} "
  "${CMAKE_CXX_COMPILER_WRAPPER}")
message("    ${CMAKE_CXX_COMPILER}")
message("  CUDA Compiler      : ${CMAKE_CUDA_COMPILER}")
message("  CUDA Compiler exec : ${CUDA_NVCC_EXECUTABLE}")
message("  CUDA Compile flags : ${CMAKE_CUDA_FLAGS}")
message("  CUDA toolkit inc   : ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES}")
message("")
message("  Python3_found       : ${Python3_FOUND}")
message("  Python3 version     : ${Python3_VERSION}")
message("  Python3 interpreter : ${Python3_EXECUTABLE}")
message("  Python3 interp. Id  : ${Python3_INTERPRETER_ID}")
message("  Python3_INCLUDE_DIRS: ${Python3_INCLUDE_DIRS}")
message("  Python3_LIBRARY_DIRS: ${Python3_LIBRARY_DIRS}")
