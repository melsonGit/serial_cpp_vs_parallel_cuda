#include "../inc/TwoDConvolution.h"
#include "../inc/MaskAttributes.h"

#include <cassert>
#include <iostream>
#include <vector>

// Reminder: When we switch from using native arrays to 2d vectors, Remove this
const std::size_t TwoDConvolution::tempConSizeInit()
{
	// Return values represent true native vector size i.e 4096^2 = 16777216
	// So our container size is really 4096

	switch (this->mTCOutputVec.size())
	{
	case 16777216: {return 4096;  break; }
	case 26214400: {return 5120; break; }
	case 37748736: {return 6144; break; }
	case 67108864: {return 8192; break; }
	case 104857600: {return 10240; break; }
	default: break;
	}

	return -1;
}
void TwoDConvolution::setContainer(const int& userInput)
{
	this->updateEventHandler(EventDirectives::populateContainer);

	// Users are displayed options 1 - 5 which translates to 0 - 4 for indexing
	int actualIndex{ userInput - 1 };

	this->processContainerSize(actualIndex);

	this->populateContainer(this->mTCInputVec, this->mTCMaskVec);

	this->setCurrSampleSize(actualIndex);

	this->updateEventHandler(EventDirectives::populateContainerComplete);
}
void TwoDConvolution::launchOp()
{
	this->updateEventHandler(EventDirectives::startOperation);
	this->OperationTimeHandler.resetStartTimer();

	// Radius rows/cols will determine when convolution occurs to prevent out of bound errors
	// twoConv utilises one for rows AND columns as we're dealing with a 2D mask vector
	int radiusOffsetRows{ 0 };
	int radiusOffsetCols{ 0 };

	// Replace this var with mTCOutputVec.size() in all if loop conditional statements when we use 2d vectors
	std::size_t tempConSize{ this->tempConSizeInit() };

	// Accumulate results
	std::size_t resultVar{};

	// For each row
	for (auto rowIn{ 0 }; rowIn < tempConSize; ++rowIn)
	{
		// For each column in that row
		for (auto colIn{ 0 }; colIn < tempConSize; ++colIn)
		{
			// Assign the tempResult variable a value
			resultVar = 0;

			// For each mask row
			for (auto maskRowIn{ 0 }; maskRowIn < MaskAttributes::maskDim; ++maskRowIn)
			{
				// Update offset value for row
				radiusOffsetRows = rowIn - MaskAttributes::maskOffset + maskRowIn;

				// For each mask column in that row
				for (auto maskColIn{ 0 }; maskColIn < MaskAttributes::maskDim; ++maskColIn)
				{
					// Update offset value for column
					radiusOffsetCols = colIn - MaskAttributes::maskOffset + maskColIn;

					// Range check for rows
					if (radiusOffsetRows >= 0 && radiusOffsetRows < tempConSize)
					{
						// Range check for columns
						if (radiusOffsetCols >= 0 && radiusOffsetCols < tempConSize)
						{
							// Accumulate results into resultVar
							resultVar += this->mTCInputVec[radiusOffsetRows * tempConSize + radiusOffsetCols]
								* this->mTCMaskVec[maskRowIn * MaskAttributes::maskDim + maskColIn];
						}
					}
				}
			}
		}
		// Assign resultVec the accumulated value of resultVar 
		mTCOutputVec[rowIn] = resultVar;
	}

	this->OperationTimeHandler.collectElapsedTimeData();
	this->updateEventHandler(EventDirectives::endOperation);
}
void TwoDConvolution::validateResults()
{
	this->updateEventHandler(EventDirectives::validateResults);

	// Assists in determining when convolution can occur to prevent out of bound errors
	// Used in conjunction with maskAttributes::maskOffset
	int radiusOffsetRows{ 0 };
	int radiusOffsetCols{ 0 };

	// Replace this var with mTCOutputVec.size() in all if loop conditional statements when we use 2d vectors
	std::size_t tempConSize{ this->tempConSizeInit() };

	// Accumulates our results
	std::size_t resultVar{};

	// Determines result authenticity - Assigned false value when results don't match
	bool doesMatch{ true };

	// For each row in mainVec
	for (auto rowIn{ 0 }; rowIn < tempConSize; ++rowIn)
	{
		// For each column in that row
		for (auto colIn{ 0 }; colIn < tempConSize && doesMatch; ++colIn)
		{
			// Reset resultVar to 0 on next element
			resultVar = 0;

			// For each mask row in maskVec
			for (auto maskRowIn{ 0 }; maskRowIn < MaskAttributes::maskDim; ++maskRowIn)
			{
				// Update offset value for that row
				radiusOffsetRows = rowIn - MaskAttributes::maskOffset + maskRowIn;

				// For each column in that mask row
				for (auto maskColIn{ 0 }; maskColIn < MaskAttributes::maskDim; ++maskColIn)
				{
					// Update offset value for that column
					radiusOffsetCols = colIn - MaskAttributes::maskOffset + maskColIn;

					// Check if we're hanging off mask row
					if (radiusOffsetRows >= 0 && radiusOffsetRows < tempConSize)
					{
						// Check if we're hanging off mask column
						if (radiusOffsetCols >= 0 && radiusOffsetCols < tempConSize)
						{
							// Accumulate results into resultVar
							resultVar += this->mTCInputVec[radiusOffsetRows * tempConSize + radiusOffsetCols]
								* this->mTCMaskVec[maskRowIn * MaskAttributes::maskDim + maskColIn];
						}
					}
				}
			}
		}
		// Check accumulated resultVar value with corresponding value in resultVec
		if (resultVar != this->mTCOutputVec[rowIn])
			doesMatch = false;
	}

	this->setValidationStatus(doesMatch);

	// Assert and abort when results don't match
	assert(doesMatch && "Check failed! Accumulated resultVar value doesn't match corresponding value in mTCOutputVec (twoConv).");

	this->updateEventHandler(EventDirectives::resultsValidated);
}
void TwoDConvolution::processContainerSize(const int& newIndex)
{
	// Convolution-specific mask vector size allocation - remains the same size regardless
	// If empty (first run), resize the mask vector - if already resized (second run +), ignore
	if (this->mTCMaskVec.empty())
		this->mTCMaskVec.resize(MaskAttributes::maskDim * MaskAttributes::maskDim);

	if (this->isNewContainer() || this->isContainerSmallerSize(newIndex))
		this->resizeContainer(this->mSampleSizes[newIndex], this->mTCInputVec, this->mTCOutputVec);

	else if (this->isContainerSameSize(newIndex))
		return;

	else if (this->isContainerLargerSize(newIndex))
	{
		this->resizeContainer(this->mSampleSizes[newIndex], this->mTCInputVec, this->mTCOutputVec);
		// Non-binding - IDE will decide if this will execute
		this->shrinkContainer(this->mTCInputVec, this->mTCOutputVec);
	}

	// Only set next vecIndex if current container is smaller / larger / new
	this->setVecIndex(newIndex);
}