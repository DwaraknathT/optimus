cmake_minimum_required(VERSION 3.15 FATAL_ERROR)

project(optimus VERSION 0.1 LANGUAGES CXX CUDA)

find_package(CUDA  REQUIRED)
include_directories("${CUDA_INCLUDE_DIRS}")

###########################################################

set(CXX_STD "17" CACHE STRING "C++ standard")
set(CUDA_PATH ${CUDA_TOOLKIT_ROOT_DIR})
set(CMAKE_MODULE_PATH ${PROJECT_SOURCE_DIR}/cmake/Modules)
list(APPEND CMAKE_MODULE_PATH ${CUDA_PATH}/lib64)
# setting compiler flags
set(CMAKE_C_FLAGS    "${CMAKE_C_FLAGS}")
set(CMAKE_CXX_FLAGS  "${CMAKE_CXX_FLAGS} -std=c++17")
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -Xcompiler -Wall -ldl") # -Xptxas -v

set(SM_SETS 52 60 61 70 75 80 86 90)
set(USING_WMMA False)
set(FIND_SM False)
foreach(SM_NUM IN LISTS SM_SETS)
  string(FIND "${SM}" "${SM_NUM}" SM_POS)
  if(SM_POS GREATER -1)
    set(FIND_SM True)
    set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -gencode=arch=compute_${SM_NUM},code=\\\"sm_${SM_NUM},compute_${SM_NUM}\\\"")

    if (SM_NUM STREQUAL 70 OR SM_NUM STREQUAL 75 OR SM_NUM STREQUAL 80 OR SM_NUM STREQUAL 86 OR SM_NUM STREQUAL 90)
      set(USING_WMMA True)
    endif()
    list(APPEND CMAKE_CUDA_ARCHITECTURES ${SM_NUM})
    message("-- Assign GPU architecture (sm=${SM_NUM})")
  endif()
endforeach()

if(USING_WMMA STREQUAL True)
  set(CMAKE_C_FLAGS    "${CMAKE_C_FLAGS}    -DWMMA")
  set(CMAKE_CXX_FLAGS  "${CMAKE_CXX_FLAGS}  -DWMMA")
  set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -DWMMA")
  message("-- Use WMMA")
endif()

if(NOT (FIND_SM STREQUAL True))
  set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS}  \
                        -gencode=arch=compute_70,code=\\\"sm_70,compute_70\\\" \
                        -gencode=arch=compute_75,code=\\\"sm_75,compute_75\\\" \
                        -gencode=arch=compute_80,code=\\\"sm_80,compute_80\\\" \
                        -gencode=arch=compute_86,code=\\\"sm_86,compute_86\\\" \
                        ")
  set(CMAKE_C_FLAGS    "${CMAKE_C_FLAGS}    -DWMMA")
  set(CMAKE_CXX_FLAGS  "${CMAKE_CXX_FLAGS}  -DWMMA")
  set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -DWMMA")
  set(CMAKE_CUDA_ARCHITECTURES 70 75 80 86)
  message("-- Assign GPU architecture (sm=70,75,80,86)")
endif()

set(CMAKE_C_FLAGS_DEBUG    "${CMAKE_C_FLAGS_DEBUG}    -Wall -O0")
set(CMAKE_CXX_FLAGS_DEBUG  "${CMAKE_CXX_FLAGS_DEBUG}  -Wall -O0")
set(CMAKE_CUDA_FLAGS_DEBUG "${CMAKE_CUDA_FLAGS_DEBUG} -O0 -G -Xcompiler -Wall -DCUDA_PTX_FP8_F2FP_ENABLED")

set(CMAKE_CXX_STANDARD "${CXX_STD}")
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} --expt-extended-lambda")
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} --expt-relaxed-constexpr")
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} --std=c++${CXX_STD} -DCUDA_PTX_FP8_F2FP_ENABLED")
set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} -O3")
set(CMAKE_CUDA_FLAGS_RELEASE "${CMAKE_CUDA_FLAGS_RELEASE} -Xcompiler -O3 -DCUDA_PTX_FP8_F2FP_ENABLED")

set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)

set(COMMON_HEADER_DIRS
  ${PROJECT_SOURCE_DIR}
  ${CUDA_PATH}/include
)
message("-- COMMON_HEADER_DIRS: ${COMMON_HEADER_DIRS}")

set(COMMON_LIB_DIRS
  ${CUDA_PATH}/lib64
)

option(BUILD_PYBIND "Build projects about pybind11" ON)
if(BUILD_PYBIND)
    set(CMAKE_CXX_FLAGS  "${CMAKE_CXX_FLAGS} -fPIC")
    execute_process(COMMAND ${PYTHON_PATH} "-c" "from __future__ import print_function; import os; import pybind11; print(os.path.dirname(pybind11.__file__),end='');"
                    RESULT_VARIABLE _PYTHON_SUCCESS
                    OUTPUT_VARIABLE PYBIND_DIR)
    list(APPEND CMAKE_PREFIX_PATH ${PYBIND_DIR})
    find_package(pybind11 REQUIRED)
    message(STATUS "Found pybind11 v${pybind11_VERSION}: ${pybind11_INCLUDE_DIRS}")
endif() 

include_directories(
  ${COMMON_HEADER_DIRS}
)

link_directories(
  ${COMMON_LIB_DIRS}
)

add_subdirectory(optimus)
add_subdirectory(examples)

###########################################################

add_library(optimus SHARED 
    $<TARGET_OBJECTS:memanager>
    $<TARGET_OBJECTS:log_utils>
    $<TARGET_OBJECTS:tensor>
    $<TARGET_OBJECTS:gemm_kernels>
    $<TARGET_OBJECTS:arithmetic_kernels>
)
set_property(TARGET optimus PROPERTY POSITION_INDEPENDENT_CODE ON)
set_property(TARGET optimus PROPERTY CUDA_RESOLVE_DEVICE_SYMBOLS ON)
target_link_libraries(optimus PUBLIC -lcudart -lcublas -lcublasLt -lcurand)

include(GNUInstallDirs)
set(INSTALL_CONFIGDIR ${CMAKE_INSTALL_LIBDIR}/cmake/FasterTransformer)

install(
  TARGETS
    optimus
  EXPORT
    optimus-targets
  LIBRARY DESTINATION ${CMAKE_INSTALL_PREFIX}/backends/fastertransformer
  ARCHIVE DESTINATION ${CMAKE_INSTALL_PREFIX}/backends/fastertransformer
)

install(
  EXPORT
    optimus-targets
  FILE
    OptimusTargets.cmake
  DESTINATION
    ${INSTALL_CONFIGDIR}
)


export(
  EXPORT
    optimus-targets
  FILE
    ${CMAKE_CURRENT_BINARY_DIR}/OptimusTargets.cmake
)

export(PACKAGE optimus)