#include "../../inc/matMulti/matMultiNumGen.h"

void matMultiNumGen(std::vector<std::vector<int>>& a, std::vector<std::vector<int>>& b)
{
    std::cout << "\nMatrix Multiplication: Populating input vectors.\n";

    // Re-seed rand() function for each run
    srand((unsigned int)time(NULL));

    // Loop to populate 2D vector a
    // For each row
    for (auto iRow = 0; iRow < a.size(); iRow++)
    {
        // For each column in that row
        for (auto iCol = 0; iCol < a[iRow].size(); iCol++)
        {
            // Assign random number to vector of vector of ints to columns iCol of rows iRows
            a[iRow][iCol] = rand() % 100;
            b[iRow][iCol] = rand() % 100;
        }

    }
    std::cout << "\nMatrix Multiplication: Populating complete.\n";
}