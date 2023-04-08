#pragma once

#include <numeric>
#include <vector>
#include "optimus/utils/memanager.h"

namespace optimus {

template <typename T>
class Tensor {
   private:
    T* data_;
    std::vector<int> shape_;
    std::vector<int> stride_;
    const size_t size_;
    const int ndim_;
    const MemoryType memory_type_;

   public:
    Tensor(std::vector<int> shape,
           MemoryType memory_type = MemoryType::MEMORY_CPU)
        : shape_(shape),
          ndim_(shape.size()),
          memory_type_(memory_type),
          size_(sizeof(T) * std::accumulate(shape.begin(), shape.end(), 1,
                                            std::multiplies<int>())) {
        // Set stride
        stride_.resize(shape_.size());
        stride_[shape_.size() - 1] = 1;
        for (int i = shape_.size() - 2; i >= 0; i--) {
            stride_[i] = stride_[i + 1] * shape_[i + 1];
        }
        // // Allocate mem for tensor
        // switch (memory_type) {
        //     case MemoryType::MEMORY_CPU:
        //         data_ = new T[size_];
        //         break;
        //     case MemoryType::MEMORY_CPU_PINNED:
        //         break;
        //     case MemoryType::MEMORY_GPU:
        //         break;
        // }
    }

    ~Tensor() {}

    std::vector<int> shape() { return shape_; }

    size_t size() { return size_; }

    std::vector<int> stride() { return stride_; }
};

}  // namespace optimus