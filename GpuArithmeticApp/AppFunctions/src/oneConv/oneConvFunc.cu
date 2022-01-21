#include "../../inc/oneConv/oneConvFunc.cuh"

__global__ void oneConvFunc(const int* deviceMainVec, const int* deviceMaskVec, int* deviceResVec, const int conSize) 
{
    // Global thread ID calculation
    int rowId = blockIdx.x * blockDim.x + threadIdx.x;

    // Temp values to work around device code issue
    const int maskDim { 7 };
    const int maskOffset { maskDim / 2 };

    // Calculate the starting point for the element
    int startPoint { rowId - maskOffset };

    // Go over each element of the mask
    for (auto j { 0 }; j < maskDim; ++j)
    {
        // Ignore elements that hang off (0s don't contribute)
        if (((startPoint + j) >= 0) && (startPoint + j < conSize)) 
        {
            // Collate results
            deviceResVec[rowId] += deviceMainVec[startPoint + j] * deviceMaskVec[j];
        }
    }
}