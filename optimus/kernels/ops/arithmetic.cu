#include "optimus/kernels/ops/arithmetic.h"
#include <tuple> 

namespace optimus {
namespace ops {

template<typename T> 
void invokeAddTensors(T* a, 
                      T* b, 
                      T* c, 
                      std::tuple<int> a_shape, 
                      std::tuple<int> b_shape, 
                      std::tuple<int> c_shape) {
    printf("In add function");
}
template void invokeAddTensors<float>(float* A,
                                      float* B, 
                                      float* C, 
                                      std::tuple<int> a_shape, 
                                      std::tuple<int> b_shape, 
                                      std::tuple<int> c_shape);

} // namespace ops 
} // namespace optimus 
