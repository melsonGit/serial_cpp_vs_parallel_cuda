#pragma once
#ifndef ONE_CONV_FUNC
#define ONE_CONV_FUNC

#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include "../maskAttributes/maskAttributes.h"

__global__ void oneConvFunc(const int*, const int*, int*, const int);

#endif