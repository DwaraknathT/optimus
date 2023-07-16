#pragma once

#include <optimus/tensor.h>

namespace optimus {
namespace ops {

template <typename T>
void InvokeAffineTransformation(Tensor<T> *A, Tensor<T> *B, Tensor<T> *bias,
                                Tensor<T> *C, bool use_relu = false);

}  // namespace ops
}  // namespace optimus