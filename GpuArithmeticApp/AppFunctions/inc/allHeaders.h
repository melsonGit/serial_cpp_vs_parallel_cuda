#pragma once
#ifndef ALL_HEADERS
#define ALL_HEADERS

// Core Function
#include "core/opChoice.h"

// Parallel Vector Addition
#include "vecAdd/vecAddCore.h"

// Parallel Matrix Multiplication
#include "matMulti/matMultiCore.cuh"

// Parallel 1D Convolution
#include "oneConv/oneConvCore.cuh"

// Parallel 2D Convolution
#include "twoConv/twoConvCore.h"

#endif
