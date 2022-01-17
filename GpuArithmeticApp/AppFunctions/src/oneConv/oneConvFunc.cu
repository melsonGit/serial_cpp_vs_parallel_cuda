#include "../../inc/oneConv/oneConvFunc.cuh"

__global__ void oneConvFunc(const int* mainVec, const int* maskVec, int* resVec, const int conSize, const int maskDim) 
{
    // Global thread ID calculation
    int rowId { blockIdx.x * blockDim.x + threadIdx.x };

    // maskRadius will determine when convolution occurs to prevent out of bound errors
    const int maskRadius { maskDim / 2 };

    // Calculate the starting point for the element
    int startPoint { rowId - maskRadius };

    // Go over each element of the mask
    for (auto j { 0 }; j < maskDim; ++j)
    {
        // Ignore elements that hang off (0s don't contribute)
        if (((startPoint + j) >= 0) && (startPoint + j < conSize)) 
        {
            // Collate results
            resVec[rowId] += mainVec[startPoint + j] * maskVec[j];
        }
    }
}