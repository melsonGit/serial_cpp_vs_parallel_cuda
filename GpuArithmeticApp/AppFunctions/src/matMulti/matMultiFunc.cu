#include "../../inc/matMulti/matMultiFunc.cuh"

__global__ void matMultiFunc(const int* inputVecA, const int* inputVecB, int* resultVec, const int conSize)
{
    // Compute each thread's global row and column index
    int rowId = blockIdx.y * blockDim.y + threadIdx.y;
    int colId = blockIdx.x * blockDim.x + threadIdx.x;

    // Iterate over row, and down column
    resultVec[rowId * conSize + colId] = 0;

    for (auto rowColPairId { 0 }; rowColPairId < conSize; ++rowColPairId)
    {
        // Accumulate results for a single element
        resultVec[rowId * conSize + colId] += inputVecA[rowId * conSize + rowColPairId] * inputVecB[rowColPairId * conSize + colId];
    }
}