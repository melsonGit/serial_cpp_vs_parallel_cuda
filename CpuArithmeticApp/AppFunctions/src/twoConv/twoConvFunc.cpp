#include "../../inc/twoConv/twoConvFunc.h"
#include "../../inc/maskAttributes.h"

void twoConvFunc(std::vector<int> const& mainVec, std::vector<int> const& maskVec, std::vector<int>& resVec, const int& conSize)
{
    std::cout << "\n2D Convolution: Populating complete.\n";
    std::cout << "\n2D Convolution: Starting operation.\n";

    // Radius rows/cols will determine when convolution occurs to prevent out of bound errors
    // twoConv utilises one for rows AND columns as we're dealing with a 2D mask vector
    int radiusOffsetRows { 0 };
    int radiusOffsetCols { 0 };

    // Accumulate results
    int tempResult;

    // Go over each row
    for (auto rowIn { 0 }; rowIn < conSize; ++rowIn)
    {
        // Go over each column
        for (auto colIn { 0 }; colIn < conSize; ++colIn)
        {
            // Assign the tempResult variable a value
            tempResult = 0;

            // Go over each mask row
            for (auto maskRowIn { 0 }; maskRowIn < maskAttributes::maskDim; ++maskRowIn)
            {
                // Update offset value for row
                radiusOffsetRows = rowIn - maskAttributes::maskOffset + maskRowIn;

                // Go over each mask column
                for (auto maskColIn { 0 }; maskColIn < maskAttributes::maskDim; ++maskColIn)
                {
                    // Update offset value for column
                    radiusOffsetCols = colIn - maskAttributes::maskOffset + maskColIn;

                    // Range checks if hanging off the matrix
                    if (radiusOffsetRows >= 0 && radiusOffsetRows < conSize)
                    {
                        if (radiusOffsetCols >= 0 && radiusOffsetCols < conSize)
                        {
                            // Accumulate results into resVec
                            tempResult += mainVec[radiusOffsetRows * conSize + radiusOffsetCols] * maskVec[maskRowIn * maskAttributes::maskDim + maskColIn];
                        }
                    }
                }
            }
        }
        resVec[rowIn] = tempResult;
    }
    std::cout << "\n2D Convolution: Operation complete.\n";
}