#include "../../inc/twoConv/twoConvNumGen.h"

void twoConvNumGen(int* vecToPop, const int& conSize)
{
    // Re-seed rand() function for each run
    srand((unsigned int)time(NULL));

    if (conSize > 10)
    {
        for (auto rowIn{ 0 }; rowIn < conSize; rowIn++)
        {
            for (auto colIn{ 0 }; colIn < conSize; colIn++)
            {
                vecToPop[conSize * rowIn + colIn] = rand() % 100;
            }
        }
    }
    else
    {
        for (auto rowIn{ 0 }; rowIn < conSize; rowIn++)
        {
            for (auto colIn{ 0 }; colIn < conSize; colIn++)
            {
                vecToPop[conSize * rowIn + colIn] = rand() % 10;
            }
        }
    }
}
