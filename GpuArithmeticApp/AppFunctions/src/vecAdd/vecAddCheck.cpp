#include "../../inc/vecAdd/vecAddCheck.h"

void vecAddCheck(std::vector<int> const& inputA, std::vector<int> const& inputB, std::vector<int> const& resVec, int const& conSize)
{
	std::cout << "\nVector Addition: Authenticating results.\n";

	bool doesMatch { true };

	for (auto i { 0 }; i < conSize && doesMatch; i++)
	{
		if ((inputA[i] + inputB[i]) != resVec[i])
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