#include "../../inc/matMulti/matMultiFunc.h"

void matMultiFunc(std::vector<std::vector<int>> const& matrix_a, std::vector<std::vector<int>> const& matrix_b, std::vector<std::vector<int>>&matrix_mult, matMultiConSize const conSize)
{
    std::cout << "\nMatrix Multiplication: Starting operation.\n";

    // For each row
    for (auto row = 0; row < conSize; row++) {
        // For each column
        for (auto col = 0; col < conSize; col++) {
            // For every elements in the row-column couple
            matrix_mult[row * conSize + col] = 0;
            for (auto k = 0; k < conSize; k++) {
                // Store results of a single element from matrix_a and matrix_b into a single element of matrix_mult
                matrix_mult[row * conSize + col] += matrix_a[row * conSize + k] * matrix_b[k * conSize + col];
            }
        }
    }

    std::cout << "\nMatrix Multiplication: Operation complete.\n";
}