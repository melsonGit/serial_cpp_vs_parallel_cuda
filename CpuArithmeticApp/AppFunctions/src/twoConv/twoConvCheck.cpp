#include "../../inc/twoConv/twoConvCheck.h"
#include "../../inc/maskAttributes.h"

void twoConvCheck(std::vector<int> const& mainVec, std::vector<int> const& maskVec, std::vector<int> const& resultVec, const int& conSize)
{
    std::cout << "\n2D Convolution: Authenticating results.\n\n";

    // Assists in determining when convolution can occur to prevent out of bound errors
    // Used in conjunction with maskAttributes::maskOffset
    int radiusOffsetRows { 0 };
    int radiusOffsetCols { 0 };
   
    // Accumulates our results to check against resultVec
    int resultVar {};

    // Determines result authenticity - Assigned false value when results don't match
    bool doesMatch { true };

    // For each row in mainVec
    for (auto rowIn { 0 }; rowIn < conSize; ++rowIn)
    {
        // For each column in that row
        for (auto colIn { 0 }; colIn < conSize && doesMatch; ++colIn)
        {
            // Reset resultVar to 0 on next element
            resultVar = 0;

            // For each mask row in maskVec
            for (auto maskRowIn { 0 }; maskRowIn < maskAttributes::maskDim; ++maskRowIn)
            {
                // Update offset value for that row
                radiusOffsetRows = rowIn - maskAttributes::maskOffset + maskRowIn;

                // For each column in that mask row
                for (auto maskColIn { 0 }; maskColIn < maskAttributes::maskDim; ++maskColIn)
                {
                    // Update offset value for that column
                    radiusOffsetCols = colIn - maskAttributes::maskOffset + maskColIn;

                    // Check if we're hanging off mask row
                    if (radiusOffsetRows >= 0 && radiusOffsetRows < conSize)
                    {
                        // Check if we're hanging off mask column
                        if (radiusOffsetCols >= 0 && radiusOffsetCols < conSize)
                        {
                            // Accumulate results into resultVar
                            resultVar += mainVec[radiusOffsetRows * conSize + radiusOffsetCols] * maskVec[maskRowIn * maskAttributes::maskDim + maskColIn];
                        }
                    }
                }
            }
        }
        // Check accumulated resultVar value with corresponding value in resultVec
        if (resultVar != resultVec[rowIn])
            doesMatch = false;
        else
            continue;
    }

    if (!doesMatch)
        std::cout << "2D Convolution unsuccessful: output vector data does not match the expected result.\n"
        << "Timing results will be discarded.\n\n";
    else
        std::cout << "2D Convolution successful: output vector data matches expected results.\n"
        << "Timing results will be recorded.\n\n";
}