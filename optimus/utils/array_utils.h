#pragma once

#include <tuple>

namespace optimus {

/*
Checks the broadcast-ability of two tensor shapes a and b.
We can boil down the rules to check if we can broadcast between two shapes
to the following.
    * If ndim(a) != ndim(b), shape of the one with fewer dimensions is
    padded with 1 from the left side. For eample, if shape_a = (1, 3) and
    shape_b = (32, 5, 3) then we can extend the shape_a by add 1 to the
    right side. So shape_a then becomes (1, 1, 3).
    * If the shape of array does not match in any dimension, and shape of
    one of the dimensions is 1, we can strech it to meet the dimensions of
    the other array. In the above example, we can stretch shape_a by 32 in
    dim 0, and 5 in dim 1 to perform arithmetic ops
    * If in any dim the sizes aren't equal and none is 1, then throw an error.

*/

inline int div_ceil(int numerator, int denominator) {
    std::div_t res = std::div(numerator, denominator);
    return res.rem ? (res.quot + 1) : res.quot;
}

}  // namespace optimus