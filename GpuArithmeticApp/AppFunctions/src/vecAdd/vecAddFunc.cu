#include "../../inc/vecAdd/vecAddFunc.cuh"

// CUDA kernel for vector addition || Function that computes the sum of two vectors
__global__ void vecAddFunc(const int* __restrict inputVecA, const int* __restrict inputVecB, int* __restrict resultVec, int conSize) 
{
    // Calculate global thread ID
    int tid = (blockIdx.x * blockDim.x) + threadIdx.x;

    // Boundary check
    if (tid < conSize) resultVec[tid] = inputVecA[tid] + inputVecB[tid];

}