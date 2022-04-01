#include "../../inc/matMulti/matMultiNumGen.h"

void matMultiNumGen(std::vector<std::vector<int>>& vecToPop)
{
    // Create local distribution on stack
    std::uniform_int_distribution randNum { randNumGen::minRand, randNumGen::maxRand };

    // Loop to populate 2D vector vecToPop
    // For each row
    for (auto iRow { 0 }; iRow < vecToPop.size(); ++iRow)
    {
        // For each column in that row
        for (auto iCol { 0 }; iCol < vecToPop[iRow].size(); ++iCol)
        {
            // Assign random number to vector of vector of ints to columns iCol of rows iRows
            vecToPop[iRow][iCol] = randNum(randNumGen::mersenne);
        }
    }
}