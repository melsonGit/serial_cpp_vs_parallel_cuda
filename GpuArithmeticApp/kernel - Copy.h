// CUDA kernel for vector addition || Function that computes the sum of two vectors
__global__ void vectorAdd(const int* __restrict, const int* __restrict,
    int* __restrict, int&);