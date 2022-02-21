#include "../../inc/twoConv/twoConvFunc.cuh"

__global__ void twoConvFunc(const int* __restrict mainVec, const int* __restrict maskVec, int* __restrict resultVec, const int conSize)
{
    // Calculate and assign x / y dimensional thread a global thread ID
    int gThreadRowId = blockIdx.y * blockDim.y + threadIdx.y;
    int gThreadColId = blockIdx.x * blockDim.x + threadIdx.x;

    // Temp values to work around device code issue
    const int maskDim { 7 };
    const int maskOffset { maskDim / 2 };

    // Radius rows/cols will determine when convolution occurs to prevent out of bound errors
    // twoConv utilises one for rows AND columns as we're dealing with a 2D mask vector
    int radiusOffsetRows { gThreadRowId - maskOffset };
    int radiusOffsetCols { gThreadColId - maskOffset };

    // Accumulate results
    int resultVar{};

    // For each row
    for (auto rowId { 0 }; rowId < maskDim; ++rowId)
    {
        // For each column in that row
        for (auto colId { 0 }; colId < maskDim; ++colId)
        {
            // Range check for rows
            if ((radiusOffsetRows + rowId) >= 0 && (radiusOffsetRows + rowId) < conSize)
            {
                // Range check for columns
                if ((radiusOffsetCols + colId) >= 0 && (radiusOffsetCols + colId) < conSize)
                {
                    // Accumulate results into resultVar
                    resultVar += mainVec[(radiusOffsetRows + rowId) * conSize + (radiusOffsetCols + colId)]
                                 * maskVec[rowId * maskDim + colId];
                }
            }
        }
    }
    // Assign resultVec the accumulated value of resultVar
    resultVec[gThreadRowId * conSize + gThreadColId] = resultVar;
}