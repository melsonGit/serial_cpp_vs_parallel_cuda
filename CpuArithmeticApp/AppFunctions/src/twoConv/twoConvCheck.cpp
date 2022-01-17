#include "../../inc/twoConv/twoConvCheck.h"

void twoConvCheck(std::vector<int> const& mainVec, std::vector<int> const& maskVec, std::vector<int> const& resVec, const int& conSize, const int& maskDim)
{
    std::cout << "\n2D Convolution: Authenticating results.\n\n";

    // Radius rows/cols will determine when convolution occurs to prevent out of bound errors
    // twoConv utilises one for rows AND columns as we're dealing with a 2D mask vector
    int radiusOffsetRows { 0 };
    int radiusOffsetCols { 0 };

    const int maskOffset { maskDim / 2};

    // Accumulate results
    int tempResult;

    bool doesMatch { true };

    // Go over each row
    for (auto rowIn { 0 }; rowIn < conSize; rowIn++)
    {
        // Go over each column
        for (auto colIn { 0 }; colIn < conSize && doesMatch; colIn++)
        {
            // Assign the tempResult variable a value
            tempResult = 0;

            // Go over each mask row
            for (auto maskRowIn { 0 }; maskRowIn < maskDim; maskRowIn++)
            {
                // Update offset value for row
                radiusOffsetRows = rowIn - maskOffset + maskRowIn;

                // Go over each mask column
                for (auto maskColIn { 0 }; maskColIn < maskDim; maskColIn++)
                {
                    // Update offset value for column
                    radiusOffsetCols = colIn - maskOffset + maskColIn;

                    // Range checks if hanging off the matrix
                    if (radiusOffsetRows >= 0 && radiusOffsetRows < conSize)
                    {
                        if (radiusOffsetCols >= 0 && radiusOffsetCols < conSize)
                        {
                            // Accumulate results into resVec
                            tempResult += mainVec[radiusOffsetRows * conSize + radiusOffsetCols] * maskVec[maskRowIn * maskDim + maskColIn];
                        }
                    }
                }
            }
        }

        if (resVec[rowIn] != tempResult)
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