#pragma once

#include <iostream>
#include <numeric>
#include <vector>
#include "cuda_runtime.h"
#include "optimus/utils/cuda_utils.h"
#include "optimus/utils/memanager.h"

namespace optimus {

__device__ __host__ int getOffset(const std::initializer_list<int>& index,
                                  const int* stride) {
    int offset = 0;
    int i = 0;
    for (auto it = index.begin(); it != index.end(); ++it) {
        offset += (*it) * stride[i++];
    }
    return offset;
}

template <typename T>
class Tensor {
   public:
    T* data;
    int* shape_;
    int* stride_;

    int ndim_;
    int size_;
    int n_elements_;
    MemoryType memory_type_;

   public:
    // The setShape function definition
    template <typename InputIt>
    void setShape(InputIt shape_begin, InputIt shape_end) {
        CHECK_CUDA_ERROR(
            cudaMallocManaged((void**)&shape_, sizeof(int) * ndim_));
        std::vector<int> temp_shape(shape_begin, shape_end);
        std::copy(temp_shape.begin(), temp_shape.end(), shape_);
    }

    void allocateMemory(MemoryType memory_type) {
        if (memory_type == MemoryType::MEMORY_CPU) {
            data = (T*)malloc(size_);
        } else if (memory_type == MemoryType::MEMORY_GPU) {
            CHECK_CUDA_ERROR(cudaMalloc((void**)&data, size_));
        }
    }

    void freeMemory(MemoryType memory_type) {
        if (memory_type == MemoryType::MEMORY_CPU) {
            free(data);
        } else if (memory_type == MemoryType::MEMORY_GPU) {
            CHECK_CUDA_ERROR(cudaFree(data));
        }
    }

    Tensor(const std::initializer_list<int>& shape,
           const MemoryType memory_type = MemoryType::MEMORY_CPU)
        : Tensor(std::vector<int>(shape), memory_type) {}

    Tensor(const std::vector<int>& shape,
           const MemoryType memory_type = MemoryType::MEMORY_CPU)
        : ndim_(shape.size()),
          memory_type_(memory_type),
          n_elements_(std::accumulate(shape.begin(), shape.end(), 1,
                                      std::multiplies<int>())) {
        size_ = sizeof(T) * n_elements_;

        // Set the shape in cuda managed memory.
        setShape(shape.begin(), shape.end());

        CHECK_CUDA_ERROR(
            cudaMallocManaged((void**)&stride_, sizeof(int) * ndim_));
        stride_[ndim_ - 1] = 1;
        for (int i = ndim_ - 2; i >= 0; i--) {
            stride_[i] = stride_[i + 1] * shape_[i + 1];
        }

        allocateMemory(memory_type_);
    }

    ~Tensor() {
        freeMemory(memory_type_);
        CHECK_CUDA_ERROR(cudaFree(shape_));
        CHECK_CUDA_ERROR(cudaFree(stride_));
    }

    void to(MemoryType target_memory_type) {
        if (target_memory_type != memory_type_) {
            T* new_data;

            if (target_memory_type == MemoryType::MEMORY_CPU) {
                new_data = (T*)malloc(size_);
                CHECK_CUDA_ERROR(cudaMemcpy((void*)new_data, (void*)data, size_,
                                            cudaMemcpyDeviceToHost));
            } else if (target_memory_type == MemoryType::MEMORY_GPU) {
                CHECK_CUDA_ERROR(cudaMalloc((void**)&new_data, size_));
                CHECK_CUDA_ERROR(cudaMemcpy((void*)new_data, (void*)data, size_,
                                            cudaMemcpyHostToDevice));
            }

            freeMemory(memory_type_);
            data = new_data;
            memory_type_ = target_memory_type;
        }
    }

    __device__ __host__ T get(const std::initializer_list<int>& index) const {
        return data[getOffset(index, stride_)];
    }

    __device__ __host__ void set(const std::initializer_list<int>& index,
                                 T val) {
        data[getOffset(index, stride_)] = val;
    }

    __host__ void set(const T* src_data_) {
        if (memory_type_ == MemoryType::MEMORY_CPU) {
            std::copy(src_data_, src_data_ + size_ / sizeof(T), data);
        } else if (memory_type_ == MemoryType::MEMORY_GPU) {
            CHECK_CUDA_ERROR(cudaMemcpy((void*)data, (void*)src_data_, size_,
                                        cudaMemcpyHostToDevice));
        }
    }
};

}  // namespace optimus