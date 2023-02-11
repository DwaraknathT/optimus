#pragma once 

namespace optimus {
// Ops namespace holds all math ops defined in optimus. 
namespace ops {

/* 
Invoke the general matrix multiplication kernel.
    C = alpha * (A * B) + beta * C 
    A - (M x N) , B - (N x K), C - (M x K). 
*/
template <typename T>
void InvokeGeMM(const uint32_t M, 
                const uint32_t N, 
                const uint32_t K, 
                const float alpha, 
                const float beta);


} // namespace ops  
} // namespace optimus 
