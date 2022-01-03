#include "../../inc/matMulti/matMultiNumGen.h"

void matMultiNumGen(std::vector<std::vector<int>>& vecToPop)
{
    // Re-seed rand() function for each run
    srand((unsigned int)time(NULL));

    // Loop to populate 2D vector vecToPop
    // For each row
    for (auto iRow { 0 }; iRow < vecToPop.size(); iRow++)
    {
        // For each column in that row
        for (auto iCol { 0 }; iCol < vecToPop[iRow].size(); iCol++)
        {
            // Assign random number to vector of vector of ints to columns iCol of rows iRows
            vecToPop[iRow][iCol] = rand() % 100;
        }
    }
}