#pragma once
#ifndef VEC_ADD_FUNC
#define VEC_ADD_FUNC

#include "cuda_runtime.h"
#include "device_launch_parameters.h"

__global__ void vecAddFunc(const int* __restrict, const int* __restrict, int* __restrict c, int);

#endif