#include "../../inc/vecAdd/vecAddFunc.cuh"

// CUDA kernel for vector addition || Function that computes the sum of two vectors
__global__ void vecAddFunc(const int* __restrict inputA, const int* __restrict inputB, int* __restrict resVec, int conSize) 
{
    // Calculate global thread ID
    int tid = (blockIdx.x * blockDim.x) + threadIdx.x;

    // Boundary check
    if (tid < conSize) resVec[tid] = inputA[tid] + inputB[tid];

}