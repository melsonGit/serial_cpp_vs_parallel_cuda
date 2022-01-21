#pragma once
#ifndef TWO_CONV_FUNC
#define TWO_CONV_FUNC

#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include "../maskAttributes/maskAttributes.h"

__global__ void twoConvFunc(const int*, const int*, int*, const int);

#endif