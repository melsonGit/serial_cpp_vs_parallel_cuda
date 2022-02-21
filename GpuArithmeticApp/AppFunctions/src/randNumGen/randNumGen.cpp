#include "../../inc/randNumGen/randNumGen.h"

namespace randNumGen
{
	// Our minimum and maximum range for number generation (main input vectors only)
	extern const int minRand { 1 }, maxRand { 100 };

	// Our minimum and maximum range for number generation (convolution mask vectors only)
	extern const int minMaskRand { 1 }, maxMaskRand { 10 };

	// Seed based on clock time at system start-up
	extern std::mt19937 mersenne { static_cast<std::mt19937::result_type>(std::time(nullptr)) };
}