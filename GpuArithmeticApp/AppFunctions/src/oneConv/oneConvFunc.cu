#include "../../inc/oneConv/oneConvFunc.cuh"

// 1-D convolution kernel
//  Arguments:
//      mainVec       = padded vector
//      maskVec       = convolution mask
//      resVec        = result vector
//      conSize       = number of elements in vector
//      MASK_DIM_ONE  = number of elements in the mask

__global__ void oneConvFunc(const int* mainVec, const int* maskVec, int* resVec, const int& conSize) 
{
    // Global thread ID calculation
    int rowId = blockIdx.x * blockDim.x + threadIdx.x;

    // maskRadius will determine when convolution occurs to prevent out of bound errors
    int maskRadius { MASK_ONE_DIM / 2 };

    // Calculate the starting point for the element
    int startPoint { rowId - maskRadius };

    // Temp value for calculation
    int resultVar = 0;

    // Go over each element of the mask
    for (auto j { 0 }; j < MASK_ONE_DIM; j++)
    {
        // Ignore elements that hang off (0s don't contribute)
        if (((startPoint + j) >= 0) && (startPoint + j < conSize)) 
        {
            // accumulate partial results
            resultVar += mainVec[startPoint + j] * maskVec[j];
        }
    }

    // Write-back the results
    resVec[rowId] = resultVar;
}