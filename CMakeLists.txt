cmake_minimum_required(VERSION 3.8 FATAL_ERROR) # for PyTorch extensions, version should be greater than 3.13
project(optimus LANGUAGES CXX CUDA)

find_package(CUDA 10.2 REQUIRED)

if(${CUDA_VERSION_MAJOR} VERSION_GREATER_EQUAL "11")
  add_definitions("-DENABLE_BF16")
  message("CUDA_VERSION ${CUDA_VERSION_MAJOR} is greater or equal than 11, enabling -DENABLE_BF16 flag")
endif()

set(CXX_STD "14" CACHE STRING "C++ standard")

set(CUDA_PATH ${CUDA_TOOLKIT_ROOT_DIR})

list(APPEND CMAKE_MODULE_PATH ${CUDA_PATH}/lib64)

# profiling
option(USE_NVTX "Whether or not to use nvtx" OFF)
option(BUILD_PYBIND "Build projects about pybind11" ON)

if(USE_NVTX)
  message(STATUS "NVTX is enabled.")
  add_definitions("-DUSE_NVTX")
endif()

# setting compiler flags
set(CMAKE_C_FLAGS    "${CMAKE_C_FLAGS}")
set(CMAKE_CXX_FLAGS  "${CMAKE_CXX_FLAGS}")
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS}  -Xcompiler -Wall -ldl")

set(SM_SETS 52 60 61 70 75 80 86)
set(USING_WMMA False)
set(FIND_SM False)

foreach(SM_NUM IN LISTS SM_SETS)
  string(FIND "${SM}" "${SM_NUM}" SM_POS)
  if(SM_POS GREATER -1)
    if(FIND_SM STREQUAL False)
      set(ENV{TORCH_CUDA_ARCH_LIST} "")
    endif()
    set(FIND_SM True)
    set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -gencode=arch=compute_${SM_NUM},code=\\\"sm_${SM_NUM},compute_${SM_NUM}\\\"")

    if (SM_NUM STREQUAL 70 OR SM_NUM STREQUAL 75 OR SM_NUM STREQUAL 80 OR SM_NUM STREQUAL 86)
      set(USING_WMMA True)
    endif()

    set(CMAKE_CUDA_ARCHITECTURES ${SM_NUM})
    message("-- Assign GPU architecture (sm=${SM_NUM})")
  endif()
endforeach()

if(NOT (FIND_SM STREQUAL True))
  set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS}  \
                        -gencode=arch=compute_70,code=\\\"sm_70,compute_70\\\" \
                        -gencode=arch=compute_75,code=\\\"sm_75,compute_75\\\" \
                        -gencode=arch=compute_80,code=\\\"sm_80,compute_80\\\" \
                        -gencode=arch=compute_86,code=\\\"sm_86,compute_86\\\" \
                        ")
  #                      -rdc=true")
  set(CMAKE_C_FLAGS    "${CMAKE_C_FLAGS}    -DWMMA")
  set(CMAKE_CXX_FLAGS  "${CMAKE_CXX_FLAGS}  -DWMMA")
  set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -DWMMA")
  if(BUILD_PYT)
    set(ENV{TORCH_CUDA_ARCH_LIST} "7.0;7.5;8.0;8.6")
  endif()
  set(CMAKE_CUDA_ARCHITECTURES 70 75 80 86)
  message("-- Assign GPU architecture (sm=70,75,80,86)")
endif()

set(CMAKE_C_FLAGS_DEBUG    "${CMAKE_C_FLAGS_DEBUG}    -Wall -O0")
set(CMAKE_CXX_FLAGS_DEBUG  "${CMAKE_CXX_FLAGS_DEBUG}  -Wall -O0")
set(CMAKE_CUDA_FLAGS_DEBUG "${CMAKE_CUDA_FLAGS_DEBUG} -O0 -G -Xcompiler -Wall")

set(CMAKE_CXX_STANDARD "${CXX_STD}")
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} --expt-extended-lambda")
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} --expt-relaxed-constexpr")
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} --std=c++${CXX_STD}")

set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} -O3")
# set(CMAKE_CUDA_FLAGS_RELEASE "${CMAKE_CUDA_FLAGS_RELEASE} -Xcompiler -O3 --ptxas-options=--verbose")
set(CMAKE_CUDA_FLAGS_RELEASE "${CMAKE_CUDA_FLAGS_RELEASE} -Xcompiler -O3")

set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)

set(Python3_FIND_VIRTUALENV "STANDARD")
find_package(Python3 COMPONENTS Interpreter Development)

# if(BUILD_PYBIND)
    # # -fPIC is required by pybind11
    # # https://pybind11.readthedocs.io/en/stable/basics.html#creating-bindings-for-a-simple-function
    # set(CMAKE_CXX_FLAGS  "${CMAKE_CXX_FLAGS} -fPIC")
    # execute_process(COMMAND ${PYTHON_PATH} "-c" "from __future__ import print_function; import os; import pybind11; print(os.path.dirname(pybind11.__file__),end='');"
    #                 RESULT_VARIABLE _PYTHON_SUCCESS
    #                 OUTPUT_VARIABLE PYBIND_DIR)
    # if (NOT _PYTHON_SUCCESS MATCHES 0)
    #     message(FATAL_ERROR "Pybind config Error.")
    # endif()
    # list(APPEND CMAKE_PREFIX_PATH ${PYBIND_DIR})
    # find_package(pybind11 REQUIRED)
    # message(STATUS "Found pybind11 v${pybind11_VERSION}: ${pybind11_INCLUDE_DIRS}")
    # make sure to find python from conda, if a conda virtual env is activated
# endif()

set(COMMON_HEADER_DIRS
  ${PROJECT_SOURCE_DIR}
  ${CUDA_PATH}/include
)
message("-- COMMON_HEADER_DIRS: ${COMMON_HEADER_DIRS}")

set(COMMON_LIB_DIRS
  ${CUDA_PATH}/lib64
)

set(PYTHON_PATH "python" CACHE STRING "Python path")

include_directories(
  ${COMMON_HEADER_DIRS}
)

link_directories(
  ${COMMON_LIB_DIRS}
)

add_subdirectory(optimus)

add_executable(test test.cu)
target_link_libraries(test PUBLIC memanager gemm_kernels)
target_link_libraries(test PUBLIC -lcublas -lcublasLt -lcudart -lcurand)
