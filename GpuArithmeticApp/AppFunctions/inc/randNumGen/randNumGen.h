#pragma once
#ifndef RAND_NUM_GEN
#define RAND_NUM_GEN

#include <ctime>
#include <random>

namespace randNumGen
{
	// Our minimum and maximum range for number generation (main input vectors only)
	extern const int minRand, maxRand;

	// Our minimum and maximum range for number generation (convolution mask vectors only)
	extern const int minMaskRand, maxMaskRand;

	// Seed based on clock time at system start-up
	extern std::mt19937 mersenne;
}
#endif