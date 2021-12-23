#include "../../inc/matMulti/matMultiFunc.h"

void matMultiFunc(std::vector<std::vector<int>> const& a, std::vector<std::vector<int>> const& b, 
                  std::vector<std::vector<int>> &c, matMultiConSize const conSize)
{
    std::cout << "\nMatrix Multiplication: Starting operation.\n";

    // For each row
    for (int row = 0; row < conSize; row++) {
        // For each column in that row
        for (int col = 0; col < conSize; col++) {
            // For each element in this row-column
            c[row][col] = 0;
            for (int k = 0; k < conSize; k++) {
                // Store results of a single element from a and b into a single element of c
                c[row][col] += a[row][k] * b[k][col];

                std::cout << "Input vector 1: " << a[row][k] << '\n';
                std::cout << "Input vector 2: " << b[k][col] << '\n';
                std::cout << "Output vector: " << c[row][k] << '\n';

                std::cout << "hit\n";


            }
        }
    }

    std::cout << "\nMatrix Multiplication: Operation complete.\n";
}

#if 0
// Loop to populate 2D vector a
// For each row
for (auto iRow = 0; iRow < a.size(); iRow++)
{
    // For each column in that row
    for (auto iCol = 0; iCol < a[iRow].size(); iCol++)
        // Assign random number to vector of vector of ints to columns iCol of rows iRows
        a[iRow][iCol] = rand() % 100;
}


c[row * conSize + col] = 0;
#endif