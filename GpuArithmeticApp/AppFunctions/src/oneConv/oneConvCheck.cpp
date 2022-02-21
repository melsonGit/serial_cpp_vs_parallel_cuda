#include "../../inc/oneConv/oneConvCheck.h"

void oneConvCheck(const int* mainVec, const int* maskVec, const int* resultVec, const int& conSize)
{
    std::cout << "\n1D Convolution: Authenticating results.\n\n";

    // Determines result authenticity - Assigned false value when results don't match
    bool doesMatch { true };

    // Assists in determining when convolution can occur to prevent out of bound errors
    // Used in conjunction with maskAttributes::maskOffset
    int radiusOffsetRows { 0 };

    // Accumulates our results to check against resultVec
    int resultVar{};

    // For each row
    for (auto rowId { 0 }; rowId < conSize && doesMatch; ++rowId)
    {
        radiusOffsetRows = rowId - maskAttributes::maskOffset;

        // Reset resultVar to 0 on next element
        resultVar = 0;

        // For each mask row
        for (auto maskRowId { 0 }; maskRowId < maskAttributes::maskDim; ++maskRowId)
        {
            // Check if we're hanging off mask row
            if ((radiusOffsetRows + maskRowId >= 0) && (radiusOffsetRows + maskRowId < conSize)) 
            {
                // Accumulate results into resultVar
                resultVar += mainVec[radiusOffsetRows + maskRowId] * maskVec[maskRowId];
            }
        }
        // Check accumulated resultVar value with corresponding value in resultVec
        if (resultVar != resultVec[rowId])
            doesMatch = false;
    }
    // Assert and abort when results don't match
    assert(doesMatch && "Check failed! Accumulated resultVar value doesn't match corresponding value in resultVec (oneConv).");

    if (!doesMatch)
        std::cerr << "1D Convolution unsuccessful: output vector data does not match the expected result.\n"
        << "Timing results will be discarded.\n";
    else
        std::cout << "1D Convolution successful: output vector data matches expected results.\n"
        << "Timing results will be recorded.\n";
}