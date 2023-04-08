#pragma once

#include <numeric>
#include <vector>
#include "optimus/utils/memanager.h"

namespace optimus {

template <typename T>
class Tensor {
   public:
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
        // Allocate mem for tensor
        switch (memory_type) {
            case MemoryType::MEMORY_CPU:
                data_ = new T[size_]{};
                break;
            case MemoryType::MEMORY_CPU_PINNED:
                cudaMallocHost((T**)&data_, size_);
                memset(data_, 0, size_);
                break;
            case MemoryType::MEMORY_GPU:
                cudaMalloc((T**)&data_, size_);
                cudaMemset(data_, 0, size_);
                break;
        }
    }

    ~Tensor() {
        switch (memory_type_) {
            case MemoryType::MEMORY_CPU:
                delete[] data_;
                break;
            case MemoryType::MEMORY_CPU_PINNED:
                cudaFreeHost(data_);
                break;
            case MemoryType::MEMORY_GPU:
                cudaFree(data_);
                break;
        }
    }

    void copy_data(const T* src_data_ptr) {
        switch (memory_type_) {
            case MemoryType::MEMORY_CPU:
                memcpy(data_, src_data_ptr, size_);
                break;
            case MemoryType::MEMORY_CPU_PINNED:
                memcpy(data_, src_data_ptr, size_);
                break;
            case MemoryType::MEMORY_GPU:
                cudaMemcpy((void*)data_, (void*)src_data_ptr, size_,
                           cudaMemcpyHostToDevice);
                break;
        }
    }

    T& operator[](const std::vector<int>& index) {
        int offset = 0;
        for (size_t i = 0; i < shape_.size(); i++) {
            offset += index[i] * stride_[i];
        }
        return data_[offset];
    }

    T& operator[](std::initializer_list<int> index) {
        return operator[](std::vector<int>(index));
    }

    std::vector<int> shape() { return shape_; }

    size_t size() { return size_; }

    std::vector<int> stride() { return stride_; }
};

}  // namespace optimus