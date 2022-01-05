#include "../../inc/matMulti/matMultiFunc.cuh"

__global__ void matMultiFunc(const int* inputA, const int* inputB, int* outputC, int conSize)
{
    // Compute each thread's global row and column index
    int rowID = blockIdx.y * blockDim.y + threadIdx.y;
    int colID = blockIdx.x * blockDim.x + threadIdx.x;

    // Iterate over row, and down column
    outputC[rowID * conSize + colID] = 0;
    for (auto k { 0 }; k < conSize; k++) 
    {
        // Accumulate results for a single element
        outputC[rowID * conSize + colID] += inputA[rowID * conSize + k] * inputB[k * conSize + colID];
    }
}