#include "../inc/OneDConvolution.h"
#include "../inc/MaskAttributes.h"

#include <cassert>
#include <iostream>
#include <vector>

void OneDConvolution::setContainer(const int& userInput)
{
	int actualIndex{ userInput - 1 };

	this->mOCInputVec.resize(mSampleSizes[actualIndex]);
	this->mOCOutputVec.resize(mSampleSizes[actualIndex]);

	// Convolution-specific mask vector size allocation
	this->mOCMaskVec.resize(MaskAttributes::maskDim);

	populateContainer(this->mOCInputVec, this->mOCMaskVec);
}

void OneDConvolution::launchOp()
{
    std::cout << "\n1D Convolution: Populating complete.\n";
    std::cout << "\n1D Convolution: Starting operation.\n";

	// Assists in determining when convolution can occur to prevent out of bound errors
	// Used in conjunction with maskAttributes::maskOffset
    std::size_t radiusOffsetRows{ 0 };

	// For each row in our input vector
    for (auto rowIn{ 0 }; rowIn < this->mOCOutputVec.size(); ++rowIn)
    {
		// Update offset value for that row
        radiusOffsetRows = rowIn - MaskAttributes::maskOffset;

		// For each mask row in mOCMaskVec
        for (auto maskRowIn{ 0 }; maskRowIn < MaskAttributes::maskDim; ++maskRowIn)
        {
			// Check if we're hanging off mask row
            if ((radiusOffsetRows + maskRowIn >= 0) && (radiusOffsetRows + maskRowIn < this->mOCOutputVec.size()))
            {
				// Accumulate results into resultVar
                this->mOCOutputVec[rowIn] += this->mOCInputVec[radiusOffsetRows + maskRowIn] * this->mOCMaskVec[maskRowIn];
            }
        }
    }
    std::cout << "\n1D Convolution: Operation complete.\n";
}

void OneDConvolution::validateResults()
{
	std::cout << "\n1D Convolution: Authenticating results.\n\n";

	// Assists in determining when convolution can occur to prevent out of bound errors
	// Used in conjunction with maskAttributes::maskOffset
	std::size_t radiusOffsetRows{ 0 };

	// Accumulates our results to check against resultVec
	std::size_t resultVar{0};

	// Determines result authenticity - Assigned false value when results don't match
	bool doesMatch{ true };

	for (auto rowIn{ 0 }; rowIn < this->mOCOutputVec.size() && doesMatch; ++rowIn)
	{
		// Reset resultVar to 0 on next element
		resultVar = 0;

		// Update offset value for that row
		radiusOffsetRows = rowIn - MaskAttributes::maskOffset;

		// For each mask row in mOCMaskVec
		for (auto maskRowIn{ 0 }; maskRowIn < MaskAttributes::maskDim; ++maskRowIn)
		{
			// Check if we're hanging off mask row
			if ((radiusOffsetRows + maskRowIn >= 0) && (radiusOffsetRows + maskRowIn < this->mOCOutputVec.size()))
			{
				// Accumulate results into resultVar
				resultVar += this->mOCInputVec[radiusOffsetRows + maskRowIn] * this->mOCMaskVec[maskRowIn];
			}
		}
		// Check accumulated resultVar value with corresponding value in resultVec
		if (resultVar != this->mOCOutputVec[rowIn])
			doesMatch = false;
	}
	// Assert and abort when results don't match
	assert(doesMatch && "Check failed! Accumulated resultVar value doesn't match corresponding value in resultVec (oneConv).");

	if (!doesMatch)
		std::cerr << "1D Convolution unsuccessful: output vector data does not match the expected result.\n"
		<< "Timing results will be discarded.\n\n";
	else
		std::cout << "1D Convolution successful: output vector data matches expected results.\n"
		<< "Timing results will be recorded.\n\n";
}