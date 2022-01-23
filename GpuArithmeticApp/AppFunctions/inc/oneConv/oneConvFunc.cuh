#pragma once
#ifndef ONE_CONV_FUNC
#define ONE_CONV_FUNC

#include "../maskAttributes/maskAttributes.h"

#include "cuda_runtime.h"
#include "device_launch_parameters.h"

__global__ void oneConvFunc(const int* mainVec, const int* maskVec, int* resultVec, const int conSize);

#endif