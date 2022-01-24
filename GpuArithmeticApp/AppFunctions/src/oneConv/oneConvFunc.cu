#include "../../inc/oneConv/oneConvFunc.cuh"

__global__ void oneConvFunc(const int* mainVec, const int* maskVec, int* resultVec, const int conSize) 
{
    // Calculate and assign x dimensional thread a global thread ID
    int gThreadRowId = blockIdx.x * blockDim.x + threadIdx.x;

    // Temp values to work around device code issue
    const int maskDim { 7 };
    const int maskOffset { maskDim / 2 };

    // Calculate the starting point for the element
    int startPoint { gThreadRowId - maskOffset };

    // Go over each element of the mask
    for (auto j { 0 }; j < maskDim; ++j)
    {
        // Ignore elements that hang off (0s don't contribute)
        if (((startPoint + j) >= 0) && (startPoint + j < conSize)) 
        {
            // Collate results
            resultVec[gThreadRowId] += mainVec[startPoint + j] * maskVec[j];
        }
    }
}