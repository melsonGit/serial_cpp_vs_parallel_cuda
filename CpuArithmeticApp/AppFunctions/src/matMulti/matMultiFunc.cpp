#include "../../inc/matMulti/matMultiFunc.h"

void matMultiFunc(std::vector<std::vector<int>> const& a, std::vector<std::vector<int>> const& b,
                  std::vector<std::vector<int>>& c, matMultiConSize const& numRows)
{
    std::cout << "\nMatrix Multiplication: Starting operation.\n";

    int numCols{ 2 };

    // For each row
    for (auto i{ 0 }; i < numRows; i++) 
    {
        // For each column in that row
        for (auto j{ 0 }; j < numCols; j++) 
        {
            // For each row-column combination
            for (auto k{0}; k < numCols; k++)
                c[i][j] += a[i][k] * b[k][j];       
        }
    }

    std::cout << "\nMatrix Multiplication: Operation complete.\n";
}