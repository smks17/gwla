# gwla
A linear algebra C++ header

NOTE: This header is in development and maybe it has many bugs.

# Example

```c++
#include <iostream>

#include "gwla.hpp"

int main () {
    GW::Vec2 vec1 {1,2};
    GW::Vec2 vec2 {3,4};
    std::cout << vec1 << '+' << vec2 << '=' << (vec1 + vec2) << std::endl;
    std::cout << vec1 << '.' << vec2 << '=' << (vec1.dot(vec2)) << std::endl;

    std::cout << "--------------------" << std::endl;

    GW::Mat2 mat1 {1,2,
                   3,4};
    GW::Mat2 mat2 {1,1,
                   1,1};
    std::cout << mat1 << std::endl;
    std::cout << '+'  << std::endl;
    std::cout << mat2 << std::endl;
    std::cout << '='  << std::endl;
    std::cout << (mat1 + mat2) << std::endl;

    std::cout << "--------------------" << std::endl;

    std::cout << mat1 << std::endl;
    std::cout << '.'  << std::endl;
    std::cout << mat2 << std::endl;
    std::cout << '='  << std::endl;
    std::cout << (mat1.dot(&mat2)) << std::endl;

    return 0;
}

```
