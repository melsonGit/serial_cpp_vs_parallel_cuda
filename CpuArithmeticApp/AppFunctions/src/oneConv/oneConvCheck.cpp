#include "../../inc/oneConv/oneConvCheck.h"

void oneConvCheck(std::vector<int> const& mainVec, std::vector<int> const& maskVec, std::vector<int> const& resVec, const int& conSize, const int& maskDim)
{

	std::cout << "\n1D Convolution: Authenticating results.\n\n";

	bool doesMatch { true };

	// Radius will determine when convolution occurs to prevent out of bound errors
	const int maskRadius { maskDim / 2 };
	int startPoint { 0 };
	int resultVar;

	for (auto i { 0 }; i < conSize && doesMatch; i++)
	{
		startPoint = i - maskRadius;
		resultVar = 0;

		for (auto j { 0 }; j < maskDim; j++)
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