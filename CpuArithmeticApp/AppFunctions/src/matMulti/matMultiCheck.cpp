#include "../../inc/matMulti/matMultiCheck.h"

void matMultiCheck(std::vector<std::vector<int>> const& inputVecA, std::vector<std::vector<int>> const& inputVecB, 
				   std::vector<std::vector<int>> const& resultVec, const int& numRows)
{
	std::cout << "\nMatrix Multiplication: Authenticating results.\n\n";

	// 2 columns for 2D vector
	const int numCols { 2 };

	// Accumulates our results to check against resultVec
	int resultVar {};

	// Determines result authenticity - Assigned false value when results don't match
	bool doesMatch{ true };

	// For each row in inputVecA/B
	for (auto rowIn { 0 }; rowIn < numRows; ++rowIn)
	{
		// For each column in that row
		for (auto colIn { 0 }; colIn < numCols && doesMatch; ++colIn)
		{
			// Reset resultVar to 0 on next element
			resultVar = 0;

			// For each row-column combination
			for (auto rowColIn { 0 }; rowColIn < numCols; ++rowColIn)
			{
				// Accumulate results into resultVar
				resultVar += inputVecA[rowIn][rowColIn] * inputVecB[rowColIn][colIn];
			}

			// Check accumulated resultVar value with corresponding value in resultVec
			if (resultVar != resultVec[rowIn][colIn])
				doesMatch = false;
		}
	}
	// Assert and abort when results don't match
	assert(doesMatch && "Check failed! Accumulated resultVar value doesn't match corresponding value in resultVec (matMulti).");

	if (!doesMatch)
		std::cerr << "Matrix multiplication unsuccessful: output vector data does not match expected results.\n"
		          << "Timing results will be discarded.\n\n";
	else
		std::cout << "Matrix multiplication successful: output vector data matches expected results.\n"
		          << "Timing results will be recorded.\n\n";
}