#include "../inc/OneDConvolution.h"
#include "../inc/MaskAttributes.h"

#include <cassert>
#include <iostream>
#include <vector>

void OneDConvolution::setContainer(const int& userInput)
{
	this->updateEventHandler(EventDirectives::populateContainer);

	// Users are displayed options 1 - 5 which translates to 0 - 4 for indexing
	int actualIndex{ userInput - 1 };

	this->processContainerSize(actualIndex);

	this->populateContainer(this->mOCInputVec, this->mOCMaskVec);

	this->setCurrSampleSize(actualIndex);

	this->updateEventHandler(EventDirectives::populateContainerComplete);
}
void OneDConvolution::launchOp()
{
	this->updateEventHandler(EventDirectives::startOperation);
	this->OperationTimeHandler.resetStartTimer();

	// Assists in determining when convolution can occur to prevent out of bound errors
	// Used in conjunction with maskAttributes::maskOffset
    std::size_t radiusOffsetRows{ 0 };

	// For each row in our input vector
    for (auto rowIn{ 0 }; rowIn < this->mOCOutputVec.size(); ++rowIn)
    {
		// Update offset value for that row
        radiusOffsetRows = rowIn - MaskAttributes::maskOffset;
		this->mOCOutputVec[rowIn] = 0;

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

	this->OperationTimeHandler.collectElapsedTimeData();
	this->updateEventHandler(EventDirectives::endOperation);
}
void OneDConvolution::validateResults()
{
	this->updateEventHandler(EventDirectives::validateResults);

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

	this->setValidationStatus(doesMatch);

	// Assert and abort when results don't match
	assert(doesMatch && "Check failed! Accumulated resultVar value doesn't match corresponding value in mOCOutputVec (oneConv).");

	this->updateEventHandler(EventDirectives::resultsValidated);
}
void OneDConvolution::processContainerSize(const int& newIndex)
{
	// Convolution-specific mask vector size allocation - remains the same size regardless
	// If empty (first run), resize the mask vector - if already resized (second run +), ignore
	if (this->mOCMaskVec.empty())
		this->mOCMaskVec.resize(MaskAttributes::maskDim);

	if (this->isNewContainer() || this->isContainerSmallerSize(newIndex))
		this->resizeContainer(this->mSampleSizes[newIndex], this->mOCInputVec, this->mOCOutputVec);

	else if (this->isContainerSameSize(newIndex))
		return;

	else if (this->isContainerLargerSize(newIndex))
	{
		this->resizeContainer(this->mSampleSizes[newIndex], this->mOCInputVec, this->mOCOutputVec);
		// Non-binding - IDE will decide if this will execute
		this->shrinkContainer(this->mOCInputVec, this->mOCOutputVec);
	}

	// Only set next vecIndex if current container is smaller / larger / new
	this->setVecIndex(newIndex);
}

// CUDA Specific Functions
void OneDConvolution::allocateMemToDevice()
{

}
void OneDConvolution::copyHostToDevice()
{

}
void OneDConvolution::copyDeviceToHost()
{

}
void OneDConvolution::freeDeviceData()
{

}