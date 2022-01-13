#include "../../inc/oneConv/oneConvCheck.h"

#ifndef MASK_ONE_DIM
// Number of elements in the convolution mask
#define MASK_ONE_DIM 7
#endif

void oneConvCheck(std::vector<int> const& mainVec, std::vector<int> const& maskVec, std::vector<int> const& resVec, int const& conSize)
{

	std::cout << "\n1D Convolution: Authenticating results.\n\n";

	bool doesMatch { true };

	// Radius will determine when convolution occurs to prevent out of bound errors
	int maskRadius { MASK_ONE_DIM / 2 };
	int startPoint { 0 };

	for (auto i { 0 }; i < conSize && doesMatch; i++)
	{
		startPoint = i - maskRadius;
		int resultVar { 0 };

		for (auto j { 0 }; j < MASK_ONE_DIM; j++)
		{
			if ((startPoint + j >= 0) && (startPoint + j < conSize))
			{
				resultVar += mainVec[startPoint + j] * maskVec[j];
			}
		}

		if (resultVar != resVec[i])
			doesMatch = false;
		else
			continue;
	}
	if (!doesMatch)
		std::cout << "1D Convolution unsuccessful: output vector data does not match the expected result.\n"
		<< "Timing results will be discarded.\n\n";
	else
		std::cout << "1D Convolution successful: output vector data matches expected results.\n"
		<< "Timing results will be recorded.\n\n";
}

// Implement a feature that automatically inputs successful data into an excel spreadsheet - via python script or third-party library