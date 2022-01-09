#pragma once
#ifndef ONE_CONV_FUNC
#define ONE_CONV_FUNC

#ifndef MASK_ONE_DIM
// Number of elements in the convolution mask
#define MASK_ONE_DIM 7
#endif

#include "cuda_runtime.h"
#include "device_launch_parameters.h"

__global__ void oneConvFunc(const int*, const int*, int*, const int);

#endif