#pragma once
#include <curand.h>
#include "optimus/utils/memanager.h"

namespace optimus {
namespace layers {

template <typename T>
class Dense {
   private:
    optimus::MemManager* mem_manager_;
    optimus::MemoryType mem_type_;

   public:
    T* weight_;
    T* bias_;
    size_t input_dim_;
    size_t output_dim_;
    size_t weights_size_;
    size_t bias_size_;

    // Buffers
    T* output_buffer_; 

    Dense(size_t input_dim, size_t output_dim,
          optimus::MemoryType mem_type = optimus::MEMORY_CPU)
        : input_dim_(input_dim),
          output_dim_(output_dim),
          weights_size_(sizeof(T) * input_dim * output_dim),
          bias_size_(sizeof(T) * output_dim),
          mem_type_(mem_type) {
        // Memory manager for this layer
        mem_manager_ = new optimus::MemManager();
    }

    ~Dense() { delete mem_manager_; }

    void allocate_buffers();

    void forward(T* inputs, T* output, bool use_relu = false);
};

}  // namespace layers
}  // namespace optimus