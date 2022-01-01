#include "../../inc/twoConv/twoConvFunc.h"

void twoConvFunc(std::vector<std::vector<int>> const& mainVec, std::vector<std::vector<int>> const& maskVec, std::vector<std::vector<int>>& resVec,
                 twoConvConSize const& conSize)
{
    std::cout << "\n2D Convolution: Starting operation.\n";

    // Radius rows/cols will determine when convolution occurs to prevent out of bound errors
    // twoConv utilises one for rows AND columns as we're dealing with a 2D mask vector
    int radiusOffsetRows;
    int radiusOffsetCols;

    // Go over each row
    for (auto i { 0 }; i < conSize; i++)
    {
        // Go over each column
        for (auto j { 0 }; j < conSize; j++)
        {
            // Assign the resVec variable a value
            resVec[i][j] = 0;

            // Go over each mask row
            for (auto k { 0 }; k < MASK_TWO_DIM; k++)
            {
                // Update offset value for row
                radiusOffsetRows = i - MASK_OFFSET + k;

                // Go over each mask column
                for (auto l { 0 }; l < MASK_TWO_DIM; l++)
                {
                    // Update offset value for column
                    radiusOffsetCols = j - MASK_OFFSET + l;

                    // Range checks if hanging off the matrix
                    if (radiusOffsetRows >= 0 && radiusOffsetRows < conSize)
                    {
                        if (radiusOffsetCols >= 0 && radiusOffsetCols < conSize)
                        {
                            // Accumulate results into resVec
                            resVec[i][j] += mainVec[radiusOffsetRows * conSize][radiusOffsetCols] * maskVec[k * MASK_TWO_DIM][l];
                        }
                    }
                }
            }
        }
    }
    std::cout << "\n2D Convolution: Operation complete.\n";
}