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

    int start{ 0 };

    for (auto i{ 0 }; i < this->mOCOutputVec.size(); ++i)
    {
        start = i - MaskAttributes::maskOffset;

        for (auto j{ 0 }; j < MaskAttributes::maskDim; ++j)
        {
            if ((start + j >= 0) && (start + j < this->mOCOutputVec.size()))
            {
                this->mOCOutputVec[i] += this->mOCInputVec[start + j] * this->mOCMaskVec[j];
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
	int radiusOffsetRows{ 0 };

	// Accumulates our results to check against resultVec
	auto resultVar{0};

	// Determines result authenticity - Assigned false value when results don't match
	bool doesMatch{ true };


	for (auto rowIn{ 0 }; rowIn < this->mOCOutputVec.size() && doesMatch; ++rowIn)
	{
		// Reset resultVar to 0 on next element
		resultVar = 0;

		// Update offset value for that row
		radiusOffsetRows = rowIn - MaskAttributes::maskOffset;

		// For each mask row in maskVec
		for (auto maskRowIn{ 0 }; maskRowIn < MaskAttributes::maskDim; ++maskRowIn)
		{
			// Check if we're hanging off mask row
			if ((radiusOffsetRows + maskRowIn >= 0) && (radiusOffsetRows + maskRowIn < this->mOCOutputVec.size()))
			{
				// Accumulate results into resultVar
				resultVar += mOCInputVec[radiusOffsetRows + maskRowIn] * mOCMaskVec[maskRowIn];
			}
		}
		// Check accumulated resultVar value with corresponding value in resultVec
		if (resultVar != mOCOutputVec[rowIn])
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