#include "kernel - Copy.h"
// CUDA kernel for vector addition || Function that computes the sum of two vectors
__global__ void vectorAdd(const int* __restrict a, const int* __restrict b,
    int* __restrict c, int& no_elements) {
    // Calculate global thread ID
    int tid = (blockIdx.x * blockDim.x) + threadIdx.x;

    // Boundary check
    if (tid < no_elements) c[tid] = a[tid] + b[tid];

}