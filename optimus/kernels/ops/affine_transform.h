#pragma once

namespace optimus {
namespace ops {

template <typename T>
void InvokeAffineTransformation(T *A, T *B, T *bias, T *C, const uint32_t M,
                                const uint32_t N, const uint32_t K,
                                bool use_relu = false);

}  // namespace ops
}  // namespace optimus