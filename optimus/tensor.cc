#include <vector>
#include <numeric>
#include <functional>

#include <curand.h>

#include "optimus/tensor.h"

namespace optimus
{
    template <typename T>
    Tensor<T>::Tensor(const std::vector<size_t> shape, T *data) : rank_(shape.size()),
                                                                  shape_(shape),
                                                                  data_(data),
                                                                  size_(sizeof(float) * std::accumulate(begin(shape_), end(shape_), 1, std::multiplies<>()))
    {
    }

    // Get the element at index given by vector of indices.
    template <typename T>
    T Tensor<T>::get(const std::vector<int> index)
    {
        return 0;
    }

    // Set the element at a given index.
    template <typename T>
    void Tensor<T>::set(const T val, std::vector<int> index)
    {
    }

}