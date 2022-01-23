#include "../../inc/vecAdd/vecAddCheck.h"

void vecAddCheck(std::vector<int> const& inputVecA, std::vector<int> const& inputVecB, std::vector<int> const& resultVec, int const& conSize)
{
	std::cout << "\nVector Addition: Authenticating results.\n\n";

	bool doesMatch { true };

	for (auto rowIn { 0 }; rowIn < conSize && doesMatch; ++rowIn)
	{
		if ((inputVecA[rowIn] + inputVecB[rowIn]) != resultVec[rowIn])
			doesMatch = false;
		else
			continue;
	}

	if (!doesMatch)
		std::cout << "Vector addition unsuccessful: output vector data does not match expected results.\n"
		<< "Timing results will be discarded.\n";
	else
		std::cout << "Vector addition successful: output vector data matches expected results.\n"
		<< "Timing results will be recorded.\n";
}