#include "../../inc/twoConv/twoConvNumGen.h"

void twoConvNumGen(std::vector<int>& vecToPop)
{
    if (vecToPop.size() > (maskAttributes::maskDim * maskAttributes::maskDim))
    {
        // Create local distribution on stack
        std::uniform_int_distribution randNum { randNumGen::minRand, randNumGen::maxRand };

        // Generate random numbers via Lambda C++11 function, and place into vector
        generate(vecToPop.begin(), vecToPop.end(), [&randNum]() { return randNum(randNumGen::mersenne); });
    }
    else
    {
        // Create local distribution on stack
        std::uniform_int_distribution randNum { randNumGen::minMaskRand, randNumGen::maxMaskRand };

        // Generate random numbers via Lambda C++11 function, and place into vector
        generate(vecToPop.begin(), vecToPop.end(), [&randNum]() { return randNum(randNumGen::mersenne); });
    }
}

#if 0 
// For future 2D implementation

// Loop to populate 2D vector
// For each row
for (auto iRow { 0 }; iRow < vecToPop.size(); ++iRow)
{
    // For each column in that row
    for (auto iCol { 0 }; iCol < vecToPop[iRow].size(); ++iCol)
    {
        // Assign random number to vector of vector of ints to columns iCol of rows iRows
        vecToPop[iRow][iCol] = rand() % 100;
    }
}
#endif