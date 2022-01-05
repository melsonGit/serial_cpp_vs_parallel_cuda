#pragma once
#ifndef MULTI_FUNC
#define MULTI_FUNC

#include "cuda_runtime.h"
#include "device_launch_parameters.h"

__global__ void matMultiFunc(const int*, const int*, int*, int);

#endif


