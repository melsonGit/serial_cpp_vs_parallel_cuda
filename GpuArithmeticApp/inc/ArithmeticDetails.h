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
		inline const std::string vecAddName{"Vector Addition"};
		inline constexpr std::array<std::size_t, 5> vecAddSamples{ 25000000, 35000000, 45000000, 55000000, 65000000 };
		inline constexpr bool vecAddMaskStatus{ false };
	}

	namespace MatrixMultiplicationDetails
	{
		inline const std::string matMultiName{ "Matrix Multiplication" };
		inline constexpr std::array<std::size_t, 5> matMultiSamples{ 1048576, 4194304, 9437184, 16777216, 26214400 };
		inline constexpr bool matMultiMaskStatus{ false };
	}

	namespace OneDConvolutionDetails
	{
		inline const std::string oneDConvName{ "1-D Convolution" };
		inline constexpr std::array<std::size_t, 5> oneDConvSamples{ 10000000, 25000000, 55000000, 75000000, 90000000 };
		inline constexpr bool oneDConvMaskStatus{ true };
	}

	namespace TwoDConvolutionDetails
	{
		inline const std::string twoDConvName{ "2-D Convolution" }; 
		inline constexpr std::array<std::size_t, 5> twoDConvSamples{ 16777216, 26214400, 37748736, 67108864, 104857600 };
		inline constexpr bool twoDConvMaskStatus{ true };
	}
}
#endif