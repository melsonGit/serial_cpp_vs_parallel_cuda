#include "../../inc/oneConv/oneConvCheck.h"

void oneConvCheck(std::vector<int> const& mainVec, std::vector<int> const& maskVec, std::vector<int> const& resVec,
	 oneConvConSize const& conSize)
{

	std::cout << "\n1D Convolution: Authenticating results.\n\n";

	bool doesMatch { true };

	int maskRadius { MASK_ONE_DIM / 2 };
	int start { 0 };

	for (auto i { 0 }; i < conSize && doesMatch; i++)
	{
		start = i - maskRadius;
		int resultVar { 0 };

		for (auto j { 0 }; j < MASK_ONE_DIM; j++)
		{
			if ((start + j >= 0) && (start + j < conSize))
			{
				resultVar += mainVec[start + j] * maskVec[j];
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