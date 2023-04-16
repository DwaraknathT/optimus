#pragma once
#include <iostream>
#include <numeric>
#include "cuda_runtime.h"
#include "optimus/utils/cuda_utils.h"
#include "optimus/utils/memanager.h"

namespace optimus {

template <typename T>
struct Tensor {
    T* data;
    int* shape_;
    int* stride_;

    int ndim_;
    int size_;
    int n_elements_;
    MemoryType memory_type_;

    Tensor(const std::initializer_list<int>& shape,
           const MemoryType memory_type = MemoryType::MEMORY_CPU)
        : ndim_(shape.size()),
          memory_type_(memory_type),
          n_elements_(std::accumulate(shape.begin(), shape.end(), 1,
                                      std::multiplies<int>())) {
        size_ = sizeof(T) * n_elements_;

        if (memory_type_ == MemoryType::MEMORY_CPU) {
            data = (T*)malloc(size_);

            shape_ = (int*)malloc(sizeof(int) * ndim_);
            memcpy(shape_, shape.begin(), sizeof(int) * ndim_);

            stride_ = (int*)malloc(sizeof(int) * ndim_);
            stride_[ndim_ - 1] = 1;
            for (int i = ndim_ - 2; i >= 0; i--) {
                stride_[i] = stride_[i + 1] * shape_[i + 1];
            }

        } else if (memory_type_ == MemoryType::MEMORY_GPU) {
            cudaMallocHost((void**)&data, size_);

            int shape_buffer[ndim_];
            memcpy((void*)shape_buffer, (void*)shape.begin(),
                   sizeof(int) * ndim_);
            cudaMallocHost((void**)&shape_, sizeof(int) * ndim_);
            cudaMemcpy((void*)shape_, (void*)shape_buffer, sizeof(int) * ndim_,
                       cudaMemcpyHostToDevice);

            int stride_buffer[ndim_];
            stride_buffer[ndim_ - 1] = 1;
            for (int i = ndim_ - 2; i >= 0; i--) {
                stride_buffer[i] = stride_buffer[i + 1] * shape_buffer[i + 1];
            }
            cudaMallocHost((void**)&stride_, sizeof(int) * shape.size());
            cudaMemcpy((void*)stride_, (void*)stride_buffer,
                       sizeof(int) * ndim_, cudaMemcpyHostToDevice);
        }
    }

    ~Tensor() {
        if (memory_type_ == MemoryType::MEMORY_CPU) {
            free(data);
            free(shape_);
            free(stride_);
        } else if (memory_type_ == MemoryType::MEMORY_GPU) {
            cudaFree(data);
            cudaFreeHost(shape_);
            cudaFreeHost(stride_);
        }
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
        cudaMemcpy((void*)data, (void*)src_data_, size_,
                   cudaMemcpyHostToDevice);
    }
};

}  // namespace optimus