#include "../inc/MaskAttributes.h"

namespace MaskAttributes
{
	// 1D/2D convolution mask dimensions
	extern const int maskDim{ 7 };

	// 1D/2D convolution mask offset
	// Used to prevent out of bound errors by determining when convolution should occur
	extern const int maskOffset{ maskDim / 2 };
}