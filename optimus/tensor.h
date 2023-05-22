#pragma once
#include <iostream>
#include <numeric>
#include "cuda_runtime.h"
#include "optimus/utils/cuda_utils.h"
#include "optimus/utils/memanager.h"

namespace optimus {

class MemoryManaged {
   public:
    void* operator new(size_t len) {
        void* ptr;
        CHECK_CUDA_ERROR(cudaMallocManaged(&ptr, len));
        CHECK_CUDA_ERROR(cudaDeviceSynchronize());
        return ptr;
    }

    void operator delete(void* ptr) {
        CHECK_CUDA_ERROR(cudaDeviceSynchronize());
        CHECK_CUDA_ERROR(cudaFree(ptr));
    }
};

template <typename T>
class Tensor : public MemoryManaged {
   public:
    T* data;
    int* shape_;
    int* stride_;

    int ndim_;
    int size_;
    int n_elements_;
    MemoryType memory_type_;

   public:
    Tensor(const std::initializer_list<int>& shape,
           const MemoryType memory_type = MemoryType::MEMORY_CPU)
        : ndim_(shape.size()),
          memory_type_(memory_type),
          n_elements_(std::accumulate(shape.begin(), shape.end(), 1,
                                      std::multiplies<int>())) {
        size_ = sizeof(T) * n_elements_;

        CHECK_CUDA_ERROR(
            cudaMallocManaged((void**)&shape_, sizeof(int) * ndim_));
        memcpy((void*)shape_, (void*)shape.begin(), sizeof(int) * ndim_);

        CHECK_CUDA_ERROR(
            cudaMallocManaged((void**)&stride_, sizeof(int) * ndim_));
        stride_[ndim_ - 1] = 1;
        for (int i = ndim_ - 2; i >= 0; i--) {
            stride_[i] = stride_[i + 1] * shape_[i + 1];
        }

        if (memory_type_ == MemoryType::MEMORY_CPU) {
            data = (T*)malloc(size_);
        } else if (memory_type_ == MemoryType::MEMORY_GPU) {
            CHECK_CUDA_ERROR(cudaMallocHost((void**)&data, size_));
        }
    }

    ~Tensor() {
        if (memory_type_ == MemoryType::MEMORY_CPU) {
            free(data);
        } else if (memory_type_ == MemoryType::MEMORY_GPU) {
            CHECK_CUDA_ERROR(cudaFree(data));
        }

        CHECK_CUDA_ERROR(cudaFree(shape_));
        CHECK_CUDA_ERROR(cudaFree(stride_));
    }

    __device__ __host__ int getOffset(const std::initializer_list<int>& index) {
        int offset = 0;
        int i = 0;
        for (auto it = index.begin(); it != index.end(); ++it) {
            offset += (*it) * stride_[i++];
        }
        return offset;
    }

    __device__ __host__ T get(const std::initializer_list<int>& index) {
        return data[getOffset(index)];
    }

    __device__ __host__ void set(const std::initializer_list<int>& index,
                                 T val) {
        data[getOffset(index)] = val;
    }

    __device__ __host__ void set(T* src_data_) {
        if (memory_type_ == MemoryType::MEMORY_CPU) {
            memcpy((void*)data, (void*)src_data_, size_);
        } else if (memory_type_ == MemoryType::MEMORY_GPU) {
            CHECK_CUDA_ERROR(cudaMemcpy((void*)data, (void*)src_data_, size_,
                                        cudaMemcpyHostToDevice));
        }
    }
};

}  // namespace optimus