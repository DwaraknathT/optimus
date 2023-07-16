#pragma once

namespace optimus {
// Ops namespace holds all math ops defined in optimus.
namespace ops {

template <typename T>
void InvokeGeMM(T *A, T *B, T *C, const uint32_t M, const uint32_t N,
                const uint32_t K, const float alpha, const float beta);

}  // namespace ops
}  // namespace optimus