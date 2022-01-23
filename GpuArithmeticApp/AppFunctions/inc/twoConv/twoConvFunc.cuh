#pragma once
#ifndef TWO_CONV_FUNC
#define TWO_CONV_FUNC

#include "../maskAttributes/maskAttributes.h"

#include "cuda_runtime.h"
#include "device_launch_parameters.h"

__global__ void twoConvFunc(const int* mainVec, const int* maskVec, int* resultVec, const int conSize);

#endif