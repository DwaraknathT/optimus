#pragma once

#include "optimus/tensor.h"

namespace optimus {
namespace ops {

template <typename T>
void InvokeGeMM(Tensor<T> *A, Tensor<T> *B, Tensor<T> *C, const float alpha,
                const float beta);

}  // namespace ops
}  // namespace optimus