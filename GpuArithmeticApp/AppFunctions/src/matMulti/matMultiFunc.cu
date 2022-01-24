#include "../../inc/matMulti/matMultiFunc.cuh"

__global__ void matMultiFunc(const int* inputVecA, const int* inputVecB, int* resultVec, const int conSize)
{
    // Calculate and assign x / y dimensional thread a global thread ID
    int gThreadRowId = blockIdx.y * blockDim.y + threadIdx.y;
    int gThreadColId = blockIdx.x * blockDim.x + threadIdx.x;

    // Iterate over row, and down column
    resultVec[gThreadRowId * conSize + gThreadColId] = 0;

    // Above gThreadRow/ColId calculation skips traversal of row and col as it has already been calculated
    // This allows us to start straight at the rowColPairId
    for (auto rowColPairId { 0 }; rowColPairId < conSize; ++rowColPairId)
    {
        // Accumulate results into resultVec
        resultVec[gThreadRowId * conSize + gThreadColId] += inputVecA[gThreadRowId * conSize + rowColPairId] 
                                                          * inputVecB[rowColPairId * conSize + gThreadColId];
    }
}