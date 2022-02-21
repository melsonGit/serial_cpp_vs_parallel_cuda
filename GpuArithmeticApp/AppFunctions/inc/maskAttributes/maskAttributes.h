#pragma once
#ifndef MASK_ATTRIBUTES
#define MASK_ATTRIBUTES

#include "cuda_runtime.h"
#include "device_launch_parameters.h"

namespace maskAttributes
{
	// 1D/2D convolution mask dimensions
	extern const int maskDim;

	// 1D/2D convolution mask offset
	// Used to prevent out of bound errors by determining when convolution should occur
	extern const int maskOffset;
}
#endif