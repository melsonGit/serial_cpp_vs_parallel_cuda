#include "../inc/TwoDConvolution.h"
#include "../inc/MaskAttributes.h"

#include <cassert>
#include <iostream>
#include <vector>

void TwoDConvolution::setContainer(const int& userInput)
{
	// Users are displayed options 1 - 5 which translates to 0 - 4 for indexing
	int actualIndex{ userInput - 1 };
	// First run check - any number outside 0 - 6 is fine but just to be safe
	constexpr int firstRun{ 99 };

	// Convolution-specific mask vector size allocation - remains the same size regardless
	// If empty (first run), resize the mask vector - if already resized (second run), ignore
	if (mTCMaskVec.empty())
		this->mTCMaskVec.resize(MaskAttributes::maskDim * MaskAttributes::maskDim);

	// If first run - we'll re-size regardless
	if (this->getCurrentSize() == firstRun)
	{
		this->setCurrentSize(actualIndex);

		this->mTCInputVec.resize(mSampleSizes[actualIndex] * mSampleSizes[actualIndex]);
		this->mTCOutputVec.resize(mSampleSizes[actualIndex] * mSampleSizes[actualIndex]);

		populateContainer(this->mTCInputVec, this->mTCMaskVec);
	}
	else if (actualIndex == this->getCurrentSize()) // If user selected same sample size as last run - don't resize, just re-populate vectors
	{
		populateContainer(this->mTCInputVec, this->mTCMaskVec);
	}
	else if (actualIndex < this->getCurrentSize()) // If current sample selection is lower than previous run - resize() and then shrink_to_fit().
	{
		this->setCurrentSize(actualIndex);

		this->mTCInputVec.resize(mSampleSizes[actualIndex]);
		this->mTCOutputVec.resize(mSampleSizes[actualIndex]);

		// Non-binding - IDE will decide if this will execute
		this->mTCInputVec.shrink_to_fit();
		this->mTCOutputVec.shrink_to_fit();

		populateContainer(this->mTCInputVec, this->mTCMaskVec);
	}
	else // If selection is higher than last run
	{
		this->setCurrentSize(actualIndex);

		this->mTCInputVec.resize(mSampleSizes[actualIndex]);
		this->mTCOutputVec.resize(mSampleSizes[actualIndex]);

		populateContainer(this->mTCInputVec, this->mTCMaskVec);
	}
}
void TwoDConvolution::launchOp()
{
	using namespace MaskAttributes;

	std::cout << "\n2D Convolution: Populating complete.\n";
	std::cout << "\n2D Convolution: Starting operation.\n";

	// Radius rows/cols will determine when convolution occurs to prevent out of bound errors
	// twoConv utilises one for rows AND columns as we're dealing with a 2D mask vector
	int radiusOffsetRows{ 0 };
	int radiusOffsetCols{ 0 };

	// Accumulate results
	int resultVar{};

	// For each row
	for (auto rowIn{ 0 }; rowIn < this->mTCOutputVec.size(); ++rowIn)
	{
		// For each column in that row
		for (auto colIn{ 0 }; colIn < this->mTCOutputVec.size(); ++colIn)
		{
			// Assign the tempResult variable a value
			resultVar = 0;

			// For each mask row
			for (auto maskRowIn{ 0 }; maskRowIn < maskDim; ++maskRowIn)
			{
				// Update offset value for row
				radiusOffsetRows = rowIn - maskOffset + maskRowIn;

				// For each mask column in that row
				for (auto maskColIn{ 0 }; maskColIn < maskDim; ++maskColIn)
				{
					// Update offset value for column
					radiusOffsetCols = colIn - maskOffset + maskColIn;

					// Range check for rows
					if (radiusOffsetRows >= 0 && radiusOffsetRows < mTCOutputVec.size())
					{
						// Range check for columns
						if (radiusOffsetCols >= 0 && radiusOffsetCols < mTCOutputVec.size())
						{
							// Accumulate results into resultVar
							resultVar += this->mTCInputVec[radiusOffsetRows * mTCOutputVec.size()+ radiusOffsetCols]
								* this->mTCMaskVec[maskRowIn * maskDim + maskColIn];
						}
					}
				}
			}
		}
		// Assign resultVec the accumulated value of resultVar 
		mTCOutputVec[rowIn] = resultVar;
	}
	std::cout << "\n2D Convolution: Operation complete.\n";
}
void TwoDConvolution::validateResults()
{
	using namespace MaskAttributes;

	std::cout << "\n2D Convolution: Authenticating results.\n\n";

	// Assists in determining when convolution can occur to prevent out of bound errors
	// Used in conjunction with maskAttributes::maskOffset
	int radiusOffsetRows{ 0 };
	int radiusOffsetCols{ 0 };

	// Accumulates our results
	int resultVar{};

	// Determines result authenticity - Assigned false value when results don't match
	bool doesMatch{ true };

	// For each row in mainVec
	for (auto rowIn{ 0 }; rowIn < mTCOutputVec.size(); ++rowIn)
	{
		// For each column in that row
		for (auto colIn{ 0 }; colIn < mTCOutputVec.size() && doesMatch; ++colIn)
		{
			// Reset resultVar to 0 on next element
			resultVar = 0;

			// For each mask row in maskVec
			for (auto maskRowIn{ 0 }; maskRowIn < maskDim; ++maskRowIn)
			{
				// Update offset value for that row
				radiusOffsetRows = rowIn - maskOffset + maskRowIn;

				// For each column in that mask row
				for (auto maskColIn{ 0 }; maskColIn < maskDim; ++maskColIn)
				{
					// Update offset value for that column
					radiusOffsetCols = colIn - maskOffset + maskColIn;

					// Check if we're hanging off mask row
					if (radiusOffsetRows >= 0 && radiusOffsetRows < mTCOutputVec.size())
					{
						// Check if we're hanging off mask column
						if (radiusOffsetCols >= 0 && radiusOffsetCols < mTCOutputVec.size())
						{
							// Accumulate results into resultVar
							resultVar += this->mTCInputVec[radiusOffsetRows * mTCOutputVec.size() + radiusOffsetCols]
								* this->mTCMaskVec[maskRowIn * maskDim + maskColIn];
						}
					}
				}
			}
		}
		// Check accumulated resultVar value with corresponding value in resultVec
		if (resultVar != this->mTCOutputVec[rowIn])
			doesMatch = false;
	}
	// Assert and abort when results don't match
	assert(doesMatch && "Check failed! Accumulated resultVar value doesn't match corresponding value in mTCOutputVec (twoConv).");

	if (!doesMatch)
		std::cerr << "2D Convolution unsuccessful: output vector data does not match the expected result.\n"
		<< "Timing results will be discarded.\n\n";
	else
		std::cout << "2D Convolution successful: output vector data matches expected results.\n"
		<< "Timing results will be recorded.\n\n";
}