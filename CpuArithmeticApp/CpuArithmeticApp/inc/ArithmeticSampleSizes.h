#pragma once
#ifndef ARITHMETIC_SAMPLE_SIZES
#define ARITHMETIC_SAMPLE_SIZES

#include <array>

// Sample sizes for each operation managed in this file

namespace ArithmeticSampleSizes
{
	namespace VectorAdditionSamples
	{
		inline constexpr std::array<unsigned int, 5> vecAddSamples{ 25000000, 35000000, 45000000, 55000000, 65000000 };
	}

	namespace MatrixMultiplicationSamples
	{
		inline constexpr std::array<unsigned int, 5> matMultiSamples{ 1048576, 4194304, 9437184, 16777216, 26214400 };
	}

	namespace OneDConvolutionSamples
	{
		inline constexpr std::array<unsigned int, 5> oneConvamples{ 10000000, 25000000, 55000000, 75000000, 90000000 };
	}

	namespace TwoDConvolutionSamples
	{
		inline constexpr std::array<unsigned int, 5> twoConvSamples{ 16777216, 26214400, 37748736, 67108864, 104857600 };
	}
}
#endif