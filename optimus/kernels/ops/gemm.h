#pragma once 

namespace optimus {
    // The following namespace holds all ops defined in optimus. 
    namespace ops {
        // Function to invoke a general matrix multiplication kernel.
        int InvokeGeMM(int a, int b);
    }
}
