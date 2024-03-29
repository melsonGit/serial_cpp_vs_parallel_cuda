#pragma once
#ifndef TWO_CONV_FUNC
#define TWO_CONV_FUNC

#include "../maskAttributes/maskAttributes.h"

#include "cuda_runtime.h"
#include "device_launch_parameters.h"

__global__ void twoConvFunc(const int* __restrict mainVec, const int* __restrict maskVec, int* __restrict resultVec, const int conSize);

#endif