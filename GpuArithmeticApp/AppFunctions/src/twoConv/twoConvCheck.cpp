#include "../../inc/twoConv/twoConvCheck.h"

void twoConvCheck(const int* mainVec, const int* maskVec, const int* resultVec, const int& conSize)
{
    std::cout << "\n2D Convolution: Authenticating results.\n\n";

    // Determines result authenticity - Assigned false value when results don't match
    bool doesMatch { true };

    // Assists in determining when convolution can occur to prevent out of bound errors
    // Used in conjunction with maskAttributes::maskOffset
    int radiusOffsetRows { 0 };
    int radiusOffsetCols { 0 };

    // Accumulates our results to check against resultVec
    int resultVar{};

    // For each row
    for (auto rowId { 0 }; rowId < conSize; ++rowId)
    {
        // For each column in that row
        for (auto colId { 0 }; colId < conSize && doesMatch; ++colId)
        {
            // Reset resultVar to 0 on next element
            resultVar = 0;

            // For each mask row
            for (auto maskRowId { 0 }; maskRowId < maskAttributes::maskDim; ++maskRowId)
            {
                // Update offset value for row
                radiusOffsetRows = rowId - maskAttributes::maskOffset + maskRowId;

                // For each mask column in that mask row
                for (auto maskColId { 0 }; maskColId < maskAttributes::maskDim; ++maskColId)
                {
                    // Update offset value for column
                    radiusOffsetCols = colId - maskAttributes::maskOffset + maskColId;

                    // Check if we're hanging off mask row
                    if (radiusOffsetRows >= 0 && radiusOffsetRows < conSize) 
                    {
                        // Check if we're hanging off mask column
                        if (radiusOffsetCols >= 0 && radiusOffsetCols < conSize) 
                        {
                            // Accumulate results into resultVar
                            resultVar += mainVec[radiusOffsetRows * conSize + radiusOffsetCols] * maskVec[maskRowId * maskAttributes::maskDim + maskColId];
                        }
                    }
                }
            }
            // Check accumulated resultVar value with corresponding value in resultVec
            if (resultVar != resultVec[rowId * conSize + colId])
                doesMatch = false;
        }
    }
    // Assert and abort when results don't match
    assert(doesMatch && "Check failed! Accumulated resultVar value doesn't match corresponding value in resultVec (twoConv).");

    if (!doesMatch)
        std::cout << "2D Convolution unsuccessful: output vector data does not match the expected result.\n"
        << "Timing results will be discarded.\n";
    else
        std::cout << "2D Convolution successful: output vector data matches expected results.\n"
        << "Timing results will be recorded.\n";
}