#include "../../inc/matMulti/matMultiFunc.h"

void matMultiFunc(std::vector<std::vector<int>> const& a, std::vector<std::vector<int>> const& b,
    std::vector<std::vector<int>>& c, matMultiConSize const& conSize)
{
    std::cout << "\nMatrix Multiplication: Starting operation.\n";

    int rowSize{ 2 };
    int count{ 0 };

    // For each row
    for (auto row{ 0 }; row < conSize; row++) 
    {
        auto k{ 0 };

        // For each column in that row
        for (auto col{ 0 }; col < rowSize; col++, k++) 
        {
            // Currently only looks at first two elements of the b vec

            c[row][k] += a[row][k] * b[k][col];

            std::cout << "Input vector 1: " << a[row][k] << '\n';
            std::cout << "Input vector 2: " << b[k][col] << '\n';
            std::cout << "Output vector: " << c[row][k] << '\n';

            std::cout << "BREAK\n";
            count++;
        }
    }

    std::cout << "\nMatrix Multiplication: Operation complete.\n";
    std::cout << count <<'\n';

}