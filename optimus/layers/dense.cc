#include "optimus/layers/dense.h"
#include "optimus/kernels/ops/affine_transform.h"

namespace optimus {
namespace layers {

template <typename T>
void optimus::layers::Dense<T>::allocate_buffers()
{
    weight_ = (T*)(mem_manager_->allocate(weights_size_, mem_type_));
    bias_ = (T*)(mem_manager_->allocate(bias_size_, mem_type_));
}

template <typename T>
void optimus::layers::Dense<T>::forward(T* input, T* output, bool use_relu) {
    // Forward function of the dense layer.

}

}  // namespace layers
}  // namespace optimus