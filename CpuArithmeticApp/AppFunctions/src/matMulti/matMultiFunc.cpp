#include "../../inc/matMulti/matMultiFunc.h"

void matMultiFunc(std::vector<std::vector<int>> const& a, std::vector<std::vector<int>> const& b, 
                  std::vector<std::vector<int>> &c, matMultiConSize const conSize)
{
    std::cout << "\nMatrix Multiplication: Starting operation.\n";

    // For each row
    for (auto row = 0; row < conSize; row++) {
        // For each column
        for (auto col = 0; col < conSize; col++) {
            // For every elements in the row-column couple
            c[row * conSize + col] = 0;
            for (auto k = 0; k < conSize; k++) {
                // Store results of a single element from a and b into a single element of c
                c[row * conSize + col] += a[row * conSize + k] * b[k * conSize + col];
            }
        }
    }

    std::cout << "\nMatrix Multiplication: Operation complete.\n";
}