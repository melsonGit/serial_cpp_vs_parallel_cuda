#include "../../inc/twoConv/twoConvFunc.cuh"

__constant__ int maskConstant[7 * 7];

__global__ void twoConvFunc(const int* deviceMainVec, int* deviceResVec, const int conSize, const int maskDim)
{
    // Calculate the global thread positions
    int rowId = blockIdx.y * blockDim.y + threadIdx.y;
    int colId = blockIdx.x * blockDim.x + threadIdx.x;

    int resultVar { 0 };
    const int maskOffset{ maskDim / 2 };

    // Starting index for calculation
    int startRowPoint { rowId - maskOffset };
    int startColPoint { colId - maskOffset };

    // Iterate over all the rows
    for (auto rowIn { 0 }; rowIn < maskDim; ++rowIn)
    {
        // Go over each column
        for (auto colIn { 0 }; colIn < maskDim; ++colIn)
        {
            // Range check for rows
            if ((startRowPoint + rowIn) >= 0 && (startRowPoint + rowIn) < conSize)
            {
                // Range check for columns
                if ((startColPoint + colIn) >= 0 && (startColPoint + colIn) < conSize)
                {
                    // Collate results
                    resultVar += deviceMainVec[(startRowPoint + rowIn) * conSize + (startColPoint + colIn)]
                                                             * maskConstant[rowIn * maskDim + colIn];
                }
            }
        }
    }

    deviceResVec[rowId * conSize + colId] = resultVar;
}