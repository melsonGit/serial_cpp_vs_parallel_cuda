#pragma once
#ifndef RAND_NUM_GEN
#define RAND_NUM_GEN

#include <ctime>
#include <random>

namespace RandNumGen
{
	// Our minimum and maximum range for number generation
	inline constexpr int minRand{ 1 }, maxRand{ 100 };

	// Our minimum and maximum range for number generation (convolution mask vectors only)
	inline constexpr int minMaskRand{ 1 }, maxMaskRand{ 10 };

	// Seed based on clock time at system start-up
	inline std::mt19937 mersenne{ static_cast<std::mt19937::result_type>(std::time(nullptr)) };
}
#endif