#pragma once
#ifndef MULTI_FUNC
#define MULTI_FUNC

#include "cuda_runtime.h"
#include "device_launch_parameters.h"

__global__ void matMultiFunc(const int* __restrict inputVecA, const int* __restrict inputVecB, int* __restrict resultVec, const int conSize);

#endif