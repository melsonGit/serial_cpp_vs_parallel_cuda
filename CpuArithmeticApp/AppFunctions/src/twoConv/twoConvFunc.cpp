#include "../../inc/twoConv/twoConvFunc.h"
#include "../../inc/maskAttributes.h"

void twoConvFunc(std::vector<int> const& mainVec, std::vector<int> const& maskVec, std::vector<int>& resultVec, const int& conSize)
{
    std::cout << "\n2D Convolution: Populating complete.\n";
    std::cout << "\n2D Convolution: Starting operation.\n";

    // Radius rows/cols will determine when convolution occurs to prevent out of bound errors
    // twoConv utilises one for rows AND columns as we're dealing with a 2D mask vector
    int radiusOffsetRows { 0 };
    int radiusOffsetCols { 0 };

    // Accumulate results
    int resultVar{};

    // For each row
    for (auto rowId { 0 }; rowId < conSize; ++rowId)
    {
        // For each column in that row
        for (auto colId { 0 }; colId < conSize; ++colId)
        {
            // Assign the tempResult variable a value
            resultVar = 0;

            // For each mask row
            for (auto maskRowId { 0 }; maskRowId < maskAttributes::maskDim; ++maskRowId)
            {
                // Update offset value for row
                radiusOffsetRows = rowId - maskAttributes::maskOffset + maskRowId;

                // For each mask column in that row
                for (auto maskColId { 0 }; maskColId < maskAttributes::maskDim; ++maskColId)
                {
                    // Update offset value for column
                    radiusOffsetCols = colId - maskAttributes::maskOffset + maskColId;

                    // Range check for rows
                    if (radiusOffsetRows >= 0 && radiusOffsetRows < conSize)
                    {
                        // Range check for columns
                        if (radiusOffsetCols >= 0 && radiusOffsetCols < conSize)
                        {
                            // Accumulate results into resultVar
                            resultVar += mainVec[radiusOffsetRows * conSize + radiusOffsetCols] * maskVec[maskRowId * maskAttributes::maskDim + maskColId];
                        }
                    }
                }
            }
        }
        // Assign resultVec the accumulated value of resultVar 
        resultVec[rowId] = resultVar;
    }
    std::cout << "\n2D Convolution: Operation complete.\n";
}