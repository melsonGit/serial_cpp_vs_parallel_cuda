#include "../inc/OneDConvolution.h"
#include "../inc/MaskAttributes.h"

#include <cassert>
#include <iostream>
#include <vector>

using namespace MaskAttributes;

void OneDConvolution::setContainer(const int& userInput)
{
	this->OperationEventHandler.processEvent();

	// Users are displayed options 1 - 5 which translates to 0 - 4 for indexing
	int actualIndex{ userInput - 1 };
	// First run check - any number outside 0 - 6 is fine but just to be safe
	constexpr int firstRun{ 99 };

	// If empty (first run), resize the mask vector - if already resized (second run), ignore
	if (this->mOCMaskVec.empty())
		this->mOCMaskVec.resize(maskDim);

	if (this->getVecIndex() == firstRun)
	{
		// If first run - we'll re-size regardless
		this->setVecIndex(actualIndex);
		this->mOCInputVec.resize(this->mSampleSizes[actualIndex]);
		this->mOCOutputVec.resize(this->mSampleSizes[actualIndex]);
	}
	else if (actualIndex < this->getVecIndex()) 
	{
		// If current sample selection is lower than previous run - resize() and then shrink_to_fit().
		this->setVecIndex(actualIndex);
		this->mOCInputVec.resize(this->mSampleSizes[actualIndex]);
		this->mOCOutputVec.resize(this->mSampleSizes[actualIndex]);
		// Non-binding - IDE will decide if this will execute
		this->mOCInputVec.shrink_to_fit();
		this->mOCOutputVec.shrink_to_fit();
	}
	else if (actualIndex > this->getVecIndex())
	{
		// If selection is higher than last run
		this->setVecIndex(actualIndex);
		this->mOCInputVec.resize(this->mSampleSizes[actualIndex]);
		this->mOCOutputVec.resize(this->mSampleSizes[actualIndex]);
	}

	// or we jump straight to populating if user selected same sample size as last run - don't resize, just re-populate vectors
	this->populateContainer(this->mOCInputVec, this->mOCMaskVec);

	this->setCurrSampleSize(actualIndex);

	this->OperationEventHandler.processEvent(); // <- This
	this->OperationEventHandler.processEvent();	// <-	   looks
	this->OperationEventHandler.processEvent();	// <-			 really
	this->OperationEventHandler.processEvent();	// <-					ugly
}
void OneDConvolution::launchOp()
{
	this->OperationEventHandler.processEvent();
	this->OperationTimer.resetStartTimer();

	// Assists in determining when convolution can occur to prevent out of bound errors
	// Used in conjunction with maskAttributes::maskOffset
    std::size_t radiusOffsetRows{ 0 };

	// For each row in our input vector
    for (auto rowIn{ 0 }; rowIn < this->mOCOutputVec.size(); ++rowIn)
    {
		// Update offset value for that row
        radiusOffsetRows = rowIn - maskOffset;
		this->mOCOutputVec[rowIn] = 0;

		// For each mask row in mOCMaskVec
        for (auto maskRowIn{ 0 }; maskRowIn < maskDim; ++maskRowIn)
        {
			// Check if we're hanging off mask row
            if ((radiusOffsetRows + maskRowIn >= 0) && (radiusOffsetRows + maskRowIn < this->mOCOutputVec.size()))
            {
				// Accumulate results into resultVar
                this->mOCOutputVec[rowIn] += this->mOCInputVec[radiusOffsetRows + maskRowIn] * this->mOCMaskVec[maskRowIn];
            }
        }
    }

	this->OperationTimer.collectElapsedTimeData();
	this->OperationEventHandler.processEvent();
}
void OneDConvolution::validateResults()
{
	this->OperationEventHandler.processEvent();

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
		radiusOffsetRows = rowIn - maskOffset;

		// For each mask row in mOCMaskVec
		for (auto maskRowIn{ 0 }; maskRowIn < maskDim; ++maskRowIn)
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

	this->setValidationStatus(doesMatch);

	// Assert and abort when results don't match
	assert(doesMatch && "Check failed! Accumulated resultVar value doesn't match corresponding value in mOCOutputVec (oneConv).");

	this->OperationEventHandler.processEvent();
}