#pragma once 

namespace optimus {
    // The following namespace holds all ops defined in optimus. 
    namespace ops {

    // Element wise addition of two tensors.
    template<typename T> 
    void invokeAdd(T* operand_one, T* operand_two, T* result);

    }
}
