#pragma once
#ifndef TWO_CONV_FUNC
#define TWO_CONV_FUNC

#ifndef MASK_TWO_DIM
// Number of elements in the convolution mask
#define MASK_TWO_DIM 7
#endif

#include "cuda_runtime.h"
#include "device_launch_parameters.h"

// Allocate mask in constant memory
__constant__ int maskConstant[7 * 7];

__global__ void twoConvFunc(const int*, int*, const int);

#endif