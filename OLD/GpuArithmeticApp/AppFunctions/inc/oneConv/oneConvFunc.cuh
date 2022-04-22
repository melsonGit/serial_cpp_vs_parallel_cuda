#pragma once
#ifndef ONE_CONV_FUNC
#define ONE_CONV_FUNC

#include "../maskAttributes/maskAttributes.h"

#include "cuda_runtime.h"
#include "device_launch_parameters.h"

__global__ void oneConvFunc(const int* __restrict mainVec, const int* __restrict maskVec, int* __restrict resultVec, const int conSize);

#endif