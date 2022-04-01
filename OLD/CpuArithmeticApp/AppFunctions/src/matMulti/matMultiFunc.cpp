#include "../../inc/matMulti/matMultiFunc.h"

void matMultiFunc(std::vector<std::vector<int>> const& inputVecA, std::vector<std::vector<int>> const& inputVecB,
                  std::vector<std::vector<int>>& resultVec, const int& numRows)
{
    std::cout << "\nMatrix Multiplication: Populating complete.\n";
    std::cout << "\nMatrix Multiplication: Starting operation.\n";

    const int numCols { 2 };

    // For each row
    for (auto i { 0 }; i < numRows; ++i) 
    {
        // For each column in that row
        for (auto j { 0 }; j < numCols; ++j) 
        {
            // For each row-column combination
            for (auto k { 0 }; k < numCols; ++k)
                resultVec[i][j] += inputVecA[i][k] * inputVecB[k][j];       
        }
    }

    std::cout << "\nMatrix Multiplication: Operation complete.\n";
}