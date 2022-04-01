#pragma once
#ifndef MASK_ATTRIBUTES
#define MASK_ATTRIBUTES

namespace maskAttributes
{
	// 1D/2D convolution mask dimensions
	inline constexpr int maskDim { 7 };

	// 1D/2D convolution mask offset
	// Used to prevent out of bound errors by determining when convolution should occur
	inline constexpr int maskOffset { maskDim / 2 };
}

#endif