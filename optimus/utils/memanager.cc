#include "optimus/utils/memanager.h"
#include <string>
#include <tuple>

namespace optimus {

inline cudaError_t checkCuda(cudaError_t result) {
#if defined(DEBUG) || defined(_DEBUG)
    if (result != cudaSuccess) {
        fprintf(stderr, "CUDA Runtime Error: %s\n", cudaGetErrorString(result));
        assert(result == cudaSuccess);
    }
#endif
    return result;
}

/*
Generates a string identifier for a given pointer.
Used in case there is no user provided name available.
*/
std::string optimus::MemManager::generateName(const void* ptr) {
    char address[256];
    sprintf(address, "%p", ptr);
    return std::string(address);
}

/*
Custom malloc that handles allocating memory on host pageable,
host pinned and device memories based on the memory type requested.
*/
void* optimus::MemManager::allocate(const size_t size, MemoryType mem_type,
                                    bool is_set_to_zero, std::string name) {
    void* mem_ptr;
    // If size = 0, then return nullptr.
    if (size == 0) {
        mem_ptr = nullptr;
    }
    // If memtype is cpu, then do normal malloc
    if (mem_type == MemoryType::MEMORY_CPU) {
        mem_ptr = (void*)malloc(size);
    } else if (mem_type == MemoryType::MEMORY_CPU_PINNED) {
        cudaError_t status = cudaMallocHost((void**)&mem_ptr, size);
        if (status != cudaSuccess)
            printf("Error allocating pinned host memory\n");
    } else if (mem_type == MemoryType::MEMORY_GPU) {
        cudaError_t status = cudaMalloc((void**)&mem_ptr, size);
        if (status != cudaSuccess) printf("Error allocating GPU memory\n");
    } else {
        // TODO: Replace this with logger.
        printf(
            "Error: memory type can only take MEMORY_CPU, MEMORY_CPU_PINNED, "
            "MEMORY_GPU.");
    }

    // If name is nor provided, then generate a name and add to map
    std::string key = (name == "") ? generateName(mem_ptr) : name;
    pointer_mapping_->insert({key, std::make_tuple(mem_ptr, size, mem_type)});
    return mem_ptr;
}

/*
Custom malloc that handles de-allocating memory on host pageable,
host pinned and device memories based on the memory type requested.
*/
void optimus::MemManager::deallocate(void** ptr, MemoryType mem_type,
                                     std::string name) {

    // If name is nor provided, then generate a name
    std::string key = (name == "") ? generateName(ptr) : name;
    if (mem_type == MemoryType::MEMORY_CPU) {
        free(ptr);
    } else if (mem_type == MemoryType::MEMORY_CPU_PINNED) {
        cudaError_t status = cudaFreeHost(ptr);
        if (status != cudaSuccess) printf("Error freeing GPU memory\n");
    } else if (mem_type == MemoryType::MEMORY_GPU) {
        cudaError_t status = cudaFree(ptr);
        if (status != cudaSuccess) printf("Error freeing GPU memory\n");
    }
}

}  // namespace optimus
