#include "../../inc/twoConv/twoConvNumGen.h"

void twoConvNumGen(int* vecToPop, const int& conSize)
{
    // Re-seed rand() function for each run
    srand((unsigned int)time(NULL));

    // Loop for mask vector
    if (conSize > 10)
    {
        // For every row
        for (auto rowId{ 0 }; rowId < conSize; ++rowId)
        {
            // For every column in that row
            for (auto colId{ 0 }; colId < conSize; ++colId)
            {
                // Generate random number and place into array
                vecToPop[conSize * rowId + colId] = rand() % 100;
            }
        }
    }
    else
    {
        // Loop for main vector
        // For every row
        for (auto rowId{ 0 }; rowId < conSize; ++rowId)
        {
            // For every column in that row
            for (auto colId{ 0 }; colId < conSize; ++colId)
            {
                // Generate random number and place into array
                vecToPop[conSize * rowId + colId] = rand() % 10;
            }
        }
    }
}