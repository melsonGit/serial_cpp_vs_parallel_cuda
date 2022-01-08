#include "../../inc/twoConv/twoConvCheck.h"

void twoConvCheck(const int* mainVec, const int* maskVec, const int* resVec, const int& conSize)
{
    std::cout << "\n2D Convolution: Authenticating results.\n\n";

    int resultVar;

    // Intermediate value for more readable code
    int radiusOffsetRows { 0 };
    int radiusOffsetCols { 0 };

    bool doesMatch { true };

    // Calculate mask radius to avoid subscript errors - determine where and when we calculate convolution
    int maskRadius { MASK_TWO_DIM / 2 };

    for (auto rowIn { 0 }; rowIn < conSize; rowIn++)
    {
        // Go over each column
        for (auto colIn { 0 }; colIn < conSize; colIn++)
        {
            // Reset the temp variable
            resultVar = 0;

            // Go over each mask row
            for (auto maskRow { 0 }; maskRow < MASK_TWO_DIM; maskRow++)
            {
                // Update offset value for row
                radiusOffsetRows = rowIn - maskRadius + maskRow;

                // Go over each mask column
                for (auto maskCol { 0 }; maskCol < MASK_TWO_DIM; maskCol++)
                {
                    // Update offset value for column
                    radiusOffsetCols = colIn - maskRadius + maskCol;

                    // Range checks if we are hanging off the matrix
                    if (radiusOffsetRows >= 0 && radiusOffsetRows < conSize) 
                    {
                        if (radiusOffsetCols >= 0 && radiusOffsetCols < conSize) 
                        {
                            // Accumulate partial results
                            resultVar += mainVec[radiusOffsetRows * conSize + radiusOffsetCols] * maskVec[maskRow * MASK_TWO_DIM + maskCol];
                        }
                    }
                }
            }
            if (resultVar != resVec[rowIn * conSize + colIn])
                doesMatch = false;
            else
                continue;
        }
    }

    if (!doesMatch)
        std::cout << "1D Convolution unsuccessful: output vector data does not match the expected result.\n"
        << "Timing results will be discarded.\n";
    else
        std::cout << "1D Convolution successful: output vector data matches expected results.\n"
        << "Timing results will be recorded.\n";
}