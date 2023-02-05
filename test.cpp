#include <iostream>
#include "optimus/kernels/ops/gemm.h"

using namespace optimus;

int main() {
    int a, b, c; // host copies of a, b, c
    // Setup input values
    a = 2;
    b = 7;
    c = optimus::ops::InvokeGeMM(a, b);
    std::cout<<c<<std::endl;
    return 0;
}
