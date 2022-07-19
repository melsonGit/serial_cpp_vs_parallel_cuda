#pragma once
#ifndef ARITHMETIC_OPERATION_INFO
#define ARITHMETIC_OPERATION_INFO

#include <array>
#include <string>

// Sample Size / Names for each operation are listed here

namespace ArithmeticDetails
{
	namespace VectorAdditionDetails
	{
		extern const std::string vecAddName;
		extern const std::array<std::size_t, 5> vecAddSamples;
		extern const bool vecAddMaskStatus;
	}

	namespace MatrixMultiplicationDetails
	{
		extern const std::string matMultiName;
		extern const std::array<std::size_t, 5> matMultiSamples;
		extern const bool matMultiMaskStatus;
	}

	namespace OneDConvolutionDetails
	{
		extern const std::string oneDConvName;;
		extern const std::array<std::size_t, 5> oneDConvSamples;
		extern const bool oneDConvMaskStatus;
	}

	namespace TwoDConvolutionDetails
	{
		extern const std::string twoDConvName;
		extern const std::array<std::size_t, 5> twoDConvSamples;
		extern const bool twoDConvMaskStatus;
	}
}
#endif