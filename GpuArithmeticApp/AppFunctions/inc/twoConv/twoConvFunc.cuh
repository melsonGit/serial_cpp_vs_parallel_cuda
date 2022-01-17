#pragma once
#ifndef TWO_CONV_FUNC
#define TWO_CONV_FUNC

#include "cuda_runtime.h"
#include "device_launch_parameters.h"

__global__ void twoConvFunc(const int*, int*, const int, const int);

#endif