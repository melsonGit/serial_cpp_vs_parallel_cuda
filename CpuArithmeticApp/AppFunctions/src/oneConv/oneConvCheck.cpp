#include "../../inc/oneConv/oneConvCheck.h"

void oneConvCheck(std::vector<int> const& mainVec, std::vector<int> const& maskVec, std::vector<int> const& resVec, 
				  oneConvConSize const& conSize)
{
#if 0
	std::cout << "\n1D Convolution: Authenticating results.\n\n";

	bool doesMatch = true;

	if (!doesMatch)
		std::cout << "1D Convolution unsuccessful: output vector data does not match the expected result.\n"
		<< "Timing results will be discarded.\n\n";
	else
		std::cout << "1D Convolution successful: output vector data matches expected results.\n"
		<< "Timing results will be recorded.\n\n";
#endif
}

// Implement a feature that automatically inputs successful data into an excel spreadsheet - via python script or third-party library