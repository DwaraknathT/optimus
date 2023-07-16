#pragma once

#include <cuda_runtime.h>
#include <iostream>
namespace optimus {

#define CHECK_CUDA_ERROR(val) check((val), #val, __FILE__, __LINE__)
template <typename T>
void check(T err, const char *const func, const char *const file,
           const int line);

#define CHECK_LAST_CUDA_ERROR() checkLast(__FILE__, __LINE__)
void checkLast(const char *const file, const int line);

}  // namespace optimus
