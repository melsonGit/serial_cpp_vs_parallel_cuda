#include "../inc/ArithmeticDetails.h"

namespace ArithmeticDetails
{
	namespace VectorAdditionDetails
	{
		extern const std::string vecAddName{ "Vector Addition" };
		extern const std::array<std::size_t, 5> vecAddSamples{ 25000000, 35000000, 45000000, 55000000, 65000000 };
		extern const bool vecAddMaskStatus{ false };
	}

	namespace MatrixMultiplicationDetails
	{
		extern const std::string matMultiName{ "Matrix Multiplication" };
		extern const std::array<std::size_t, 5> matMultiSamples{ 1048576, 4194304, 9437184, 16777216, 26214400 };
		extern const bool matMultiMaskStatus{ false };
	}

	namespace OneDConvolutionDetails
	{
		extern const std::string oneDConvName{ "1-D Convolution" };
		extern const std::array<std::size_t, 5> oneDConvSamples{ 10000000, 25000000, 55000000, 75000000, 90000000 };
		extern const bool oneDConvMaskStatus{ true };
	}

	namespace TwoDConvolutionDetails
	{
		extern const std::string twoDConvName{ "2-D Convolution" };
		extern const std::array<std::size_t, 5> twoDConvSamples{ 16777216, 26214400, 37748736, 67108864, 104857600 };
		extern const bool twoDConvMaskStatus{ true };
	}
}