#include <iostream>
#include <numeric>
#include "cuda_runtime.h"
#include "optimus/tensor.h"
#include "optimus/utils/array_utils.h"
#include "optimus/utils/cuda_utils.h"

using namespace optimus;

__global__ void setkernel(int* data, int* stride) {
    int i = threadIdx.x;
    int j = threadIdx.y;
    int k = blockIdx.x;
    int val = i * 100 + j * 10 + k;
    data[getOffset({i, j, k}, stride)] = val;
}

__global__ void myKernel(int* data, int* stride) {
    int i = threadIdx.x;
    int j = threadIdx.y;
    int k = blockIdx.x;
    printf("(%d, %d, %d) -> %d\n", i, j, k, data[getOffset({i, j, k}, stride)]);
}

int main() {

    int shape[] = {2, 3, 4};
    auto arr = new optimus::Tensor<int>({2, 3, 4}, optimus::MEMORY_GPU);

    dim3 grid(shape[2], 1);
    dim3 block(shape[0], shape[1]);
    setkernel<<<grid, block>>>(arr->data, arr->stride_);
    myKernel<<<grid, block>>>(arr->data, arr->stride_);
    cudaDeviceSynchronize();
    CHECK_LAST_CUDA_ERROR();
    
    return 0;
}
