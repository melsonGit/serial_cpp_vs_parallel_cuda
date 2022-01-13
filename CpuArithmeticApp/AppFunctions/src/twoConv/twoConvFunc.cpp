#include "../../inc/twoConv/twoConvFunc.h"

#ifndef MASK_TWO_DIM
// 7 x 7 convolutional mask
#define MASK_TWO_DIM 7
#endif

#ifndef MASK_OFFSET
// Amount the the matrix will hang over the matrix
#define MASK_OFFSET (MASK_TWO_DIM / 2)
#endif

void twoConvFunc(std::vector<int> const& mainVec, std::vector<int> const& maskVec, std::vector<int>& resVec, int const& conSize)
{
    std::cout << "\n2D Convolution: Populating complete.\n";
    std::cout << "\n2D Convolution: Starting operation.\n";

    // Radius rows/cols will determine when convolution occurs to prevent out of bound errors
    // twoConv utilises one for rows AND columns as we're dealing with a 2D mask vector
    int radiusOffsetRows { 0 };
    int radiusOffsetCols { 0 };

    // Accumulate results
    int tempResult { 0 };

    // Go over each row
    for (auto rowIn { 0 }; rowIn < conSize; rowIn++)
    {
        // Go over each column
        for (auto colIn { 0 }; colIn < conSize; colIn++)
        {
            // Assign the tempResult variable a value
            tempResult = 0;

            // Go over each mask row
            for (auto maskRowIn { 0 }; maskRowIn < MASK_TWO_DIM; maskRowIn++)
            {
                // Update offset value for row
                radiusOffsetRows = rowIn - MASK_OFFSET + maskRowIn;

                // Go over each mask column
                for (auto maskColIn { 0 }; maskColIn < MASK_TWO_DIM; maskColIn++)
                {
                    // Update offset value for column
                    radiusOffsetCols = colIn - MASK_OFFSET + maskColIn;

                    // Range checks if hanging off the matrix
                    if (radiusOffsetRows >= 0 && radiusOffsetRows < conSize)
                    {
                        if (radiusOffsetCols >= 0 && radiusOffsetCols < conSize)
                        {
                            // Accumulate results into resVec
                            tempResult += mainVec[radiusOffsetRows * conSize + radiusOffsetCols] * maskVec[maskRowIn * MASK_TWO_DIM + maskColIn];
                        }
                    }
                }
            }
        }
        resVec[rowIn] = tempResult;
    }
    std::cout << "\n2D Convolution: Operation complete.\n";
}