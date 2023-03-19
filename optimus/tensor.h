// A tensor data structure that can hold multi-dimensional
// data. It should also facilitate moving from row major to column major
// easily. Accessing elements must be abstracted away.
#pragma once
#include <cuda_fp16.h>
#include <cuda_bf16.h>

#include <vector>

namespace optimus
{
    // Tensor datatypes
    enum DataType
    {
        FLOAT32,
        FLOAT64,
    };

    template <typename T>
    struct Tensor
    {
        const int rank_;                  // Rank of the tensor
        const size_t size_;               // Size of the data in tensor in bytes
        const std::vector<size_t> shape_; // Shape of the tensor
        T *data_;

        // Constructor from data pointer
        Tensor(const std::vector<size_t> shape, T *data);
        // Get the element at an index
        T get(const std::vector<int> index);
        // Get the element at an index
        void set(const T val, const std::vector<int> index);
    };

} // namespace optimus
