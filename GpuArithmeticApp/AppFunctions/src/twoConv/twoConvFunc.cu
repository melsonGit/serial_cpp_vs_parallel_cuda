#include "../../inc/twoConv/twoConvFunc.cuh"

__global__ void twoConvFunc(const int* deviceMainVec, int* deviceResVec, const int conSize)
{
    // Calculate the global thread positions
    int rowId { blockIdx.y * blockDim.y + threadIdx.y };
    int colId { blockIdx.x * blockDim.x + threadIdx.x };

    // Calculate mask radius to avoid subscript errors - determine where and when we calculate convolution
    int maskRadius { MASK_TWO_DIM / 2 };

    int resultVar { 0 };

    // Starting index for calculation
    int startRowPoint { rowId - maskRadius };
    int startColPoint { colId - maskRadius };

    // Iterate over all the rows
    for (auto rowIn { 0 }; rowIn < MASK_TWO_DIM; rowIn++) 
    {
        // Go over each column
        for (auto colIn { 0 }; colIn < MASK_TWO_DIM; colIn++)
        {
            // Range check for rows
            if ((startRowPoint + rowIn) >= 0 && (startRowPoint + rowIn) < conSize)
            {
                // Range check for columns
                if ((startColPoint + colIn) >= 0 && (startColPoint + colIn) < conSize)
                {
                    // Collate results
                    resultVar += deviceMainVec[(startRowPoint + rowIn) * conSize + (startColPoint + colIn)]
                                                             * maskConstant[rowIn * MASK_TWO_DIM + colIn];
                }
            }
        }
        deviceResVec[rowId * conSize + colId] = resultVar;
    }
}