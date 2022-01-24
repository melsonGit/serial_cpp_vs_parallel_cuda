#include "../../inc/vecAdd/vecAddCheck.h"

void vecAddCheck(std::vector<int> const& inputVecA, std::vector<int> const& inputVecB, std::vector<int> const& resultVec, const int& conSize)
{
	std::cout << "\nVector Addition: Authenticating results.\n\n";

	// Determines result authenticity - Assigned false value when results don't match
	bool doesMatch { true };

	// For each row in inputVecA/B 
	for (auto rowId { 0 }; rowId < conSize && doesMatch; ++rowId)
	{
		// Check addition of both rows matches value in corresponding row in resultVec
		if ((inputVecA[rowId] + inputVecB[rowId]) != resultVec[rowId])
			doesMatch = false;
		else
			continue;
	}
	
	if (!doesMatch)
		std::cout << "Vector addition unsuccessful: output vector data does not match expected results.\n"
		          << "Timing results will be discarded.\n\n";
	else
		std::cout << "Vector addition successful: output vector data matches expected results.\n"
		          << "Timing results will be recorded.\n\n";
}