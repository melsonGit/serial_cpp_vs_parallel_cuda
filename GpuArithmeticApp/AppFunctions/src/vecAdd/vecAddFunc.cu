#include "../../inc/vecAdd/vecAddFunc.cuh"

__global__ void vecAddFunc(const int* __restrict inputVecA, const int* __restrict inputVecB, int* __restrict resultVec, const int conSize) 
{
    // Calculate and assign x dimensional thread a global thread ID
    int gThreadRowId = blockIdx.x * blockDim.x + threadIdx.x;

    // If the threads gThreadRowId is below conSize
    // Add contents from inputVecA and inputVecB into resultVec 
    if (gThreadRowId < conSize) 
        resultVec[gThreadRowId] = inputVecA[gThreadRowId] + inputVecB[gThreadRowId];
}