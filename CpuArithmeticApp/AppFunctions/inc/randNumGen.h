#pragma once
#ifndef RAND_NUM_GEN
#define RAND_NUM_GEN

#include <ctime>
#include <random>

namespace randNumGen
{
	// Our minimum and maximum range for number generation
	inline constexpr int minRand { 1 }, maxRand { 100 };

	// Seed based on clock time on system start-up
	inline std::mt19937 mersenne{ static_cast<std::mt19937::result_type>(std::time(nullptr)) };
}

#endif