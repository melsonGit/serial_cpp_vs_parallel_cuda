#include "../../inc/twoConv/twoConvNumGen.h"

void twoConvNumGen(int* vecToPop, const int& conSize)
{
    // Loop for main vector
    if (conSize > (maskAttributes::maskDim * maskAttributes::maskDim))
    {
        // Create local distribution on stack
        std::uniform_int_distribution randNum { randNumGen::minRand, randNumGen::maxRand };

        // For every row
        for (auto rowId { 0 }; rowId < conSize; ++rowId)
        {
            // For every column in that row
            for (auto colId { 0 }; colId < conSize; ++colId)
            {
                // Generate random number and place into array
                vecToPop[conSize * rowId + colId] = randNum(randNumGen::mersenne);
            }
        }
    }
    else
    {
        // Create local distribution on stack
        std::uniform_int_distribution randNum { randNumGen::minMaskRand, randNumGen::maxMaskRand };

        // Loop for mask vector
        // For every row
        for (auto rowId { 0 }; rowId < conSize; ++rowId)
        {
            // For every column in that row
            for (auto colId { 0 }; colId < conSize; ++colId)
            {
                // Generate random number and place into array
                vecToPop[conSize * rowId + colId] = randNum(randNumGen::mersenne);
            }
        }
    }
}