#include "../../inc/oneConv/oneConvCheck.h"
#include "../../inc/maskAttributes.h"

void oneConvCheck(std::vector<int> const& mainVec, std::vector<int> const& maskVec, std::vector<int> const& resultVec, const int& conSize)
{
	std::cout << "\n1D Convolution: Authenticating results.\n\n";

	// Assists in determining when convolution can occur to prevent out of bound errors
	// Used in conjunction with maskAttributes::maskOffset
	int radiusOffsetRows { 0 };

	// Accumulates our results to check against resultVec
	int resultVar {};

	// Determines result authenticity - Assigned false value when results don't match
	bool doesMatch { true };


	for (auto rowIn { 0 }; rowIn < conSize && doesMatch; ++rowIn)
	{
		// Reset resultVar to 0 on next element
		resultVar = 0;

		// Update offset value for that row
		radiusOffsetRows = rowIn - maskAttributes::maskOffset;

		// For each mask row in maskVec
		for (auto maskRowIn { 0 }; maskRowIn < maskAttributes::maskDim; ++maskRowIn)
		{
			// Check if we're hanging off mask row
			if ((radiusOffsetRows + maskRowIn >= 0) && (radiusOffsetRows + maskRowIn < conSize))
			{
				// Accumulate results into resultVar
				resultVar += mainVec[radiusOffsetRows + maskRowIn] * maskVec[maskRowIn];
			}
		}
		// Check accumulated resultVar value with corresponding value in resultVec
		if (resultVar != resultVec[rowIn])
			doesMatch = false;
	}
	// Assert and abort when results don't match
	assert(doesMatch && "Check failed! Accumulated resultVar value doesn't match corresponding value in resultVec (oneConv).");

	if (!doesMatch)
		std::cout << "1D Convolution unsuccessful: output vector data does not match the expected result.\n"
		<< "Timing results will be discarded.\n\n";
	else
		std::cout << "1D Convolution successful: output vector data matches expected results.\n"
		<< "Timing results will be recorded.\n\n";
}