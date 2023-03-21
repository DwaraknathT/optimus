/* Memory manager in optimus to handle device memory allocation, transfers.
In the interest of speed, our memory management design must have the following 
goals. 
    * Minimize the amount of data transfers from host to device. 
    * Use pinned memory when possible to increase data throughput.
    * Batch many small memory access requests in the kernels. 
    * Data transfers can be overlapped with kernel execution. Need 
      an aysnchronous approach for data loading. 
*/ 
#pragma once 

#include <iostream>
#include <string>
#include <tuple> 
#include <unordered_map>
#include <cuda_runtime.h> 

namespace optimus {

/* The different pools of memory which can hold variables. 
CPU memory is the host memory where usual memory allocation 
happens. GPU memory is the device global memory, we can use 
this via cuda malloc. 

Moving memory from host to device involves 2 copies, from host 
memory to pinned host memory, and then to device memory. This 
is slow compared to just placing the variables in pinned host 
memory as that would only involve 1 mem copy. 
*/
enum MemoryType {
    MEMORY_CPU, 
    MEMORY_GPU, 
    MEMORY_CPU_PINNED
};

class MemManager {
    /* A memory manager that takes care of allocations, transfers, etc. 
    */
   private: 
        // A map from variable names to their pointers. 
        std::unordered_map<std::string, std::tuple<void*, size_t, MemoryType>>* pointer_mapping_; 
        // Cuda stream to use for mem allocation. 
        cudaStream_t stream_ = 0; 

    public:
        // Constructor 
        MemManager(const std::string manager_name = "") {
            pointer_mapping_ = new std::unordered_map<std::string, std::tuple<void*, size_t, MemoryType>>();
        }
        // Destructor 
        ~MemManager(){
            // Free all items in the map and delete the map 
            while(!pointer_mapping_->empty()) {
                // Get the pointer address from the map.
                auto ptr_name = pointer_mapping_->begin()->first; 
                auto address = std::get<0>(pointer_mapping_->begin()->second);
                auto mem_type = std::get<2>(pointer_mapping_->begin()->second);
                // Call the custom free function.
                deallocate((void**)(address), mem_type, ptr_name);
                // Pop the deallovated ptr from map. 
                pointer_mapping_->erase(ptr_name);
            }
            delete pointer_mapping_; 
        }    
        // Function to generate a string identifier for a pointer if non is provided.
        std::string generateName(const void* ptr);
        // Function to allocate memory in pageable, pinned host, and device memories and a name to address that memory chunk.
        void* allocate(const size_t size, MemoryType mem_type, bool is_set_to_zero = false, std::string name = ""); 
        // Custom free function
        void deallocate(void** ptr, MemoryType mem_type, std::string name = "");

};

} // namespace optimus 
