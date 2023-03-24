// Basic arithmetic ops
#pragma once

#include <tuple>

namespace optimus {
namespace ops {

template <typename T>
void invokeAddTensors(T* a, T* b, T* c, std::tuple<int> a_shape,
                      std::tuple<int> b_shape, std::tuple<int> c_shape);

}  // namespace ops
}  // namespace optimus
