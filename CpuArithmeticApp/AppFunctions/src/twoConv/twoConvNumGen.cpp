#include "../../inc/twoConv/twoConvNumGen.h"

void twoConvNumGen(std::vector<int>& vecToPop)
{
    // Re-seed rand() function for each run
    srand((unsigned int)time(NULL));

    // Generate random numbers via Lambda C++11 function, and place into vector
    std::generate(vecToPop.begin(), vecToPop.end(), []() { return rand() % 100; });
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