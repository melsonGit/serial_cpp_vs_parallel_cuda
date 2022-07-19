#include "../inc/OneDConvolution.cuh"
#include "../inc/MaskAttributes.h"

#include <cassert>
#include <iostream>
#include <vector>

__global__ void oneConvKernel(const std::size_t* __restrict mainVec, const std::size_t* __restrict maskVec, std::size_t* __restrict resultVec, const std::size_t conSize)
{
	// Calculate and assign x dimensional thread a global thread ID
	int gThreadRowId = blockIdx.x * blockDim.x + threadIdx.x;

	// Temp values to work around device code issue
	const int maskDim{ 7 };
	const int maskOffset{ maskDim / 2 };

	// Calculate the starting point for the element
	int radiusOffsetRows{ gThreadRowId - maskOffset };

	// Go over each element of the mask
	for (auto maskRowIn{ 0 }; maskRowIn < maskDim; ++maskRowIn)
	{
		// Ignore elements that hang off (0s don't contribute)
		if (((radiusOffsetRows + maskRowIn) >= 0) && (radiusOffsetRows + maskRowIn < conSize))
		{
			// Collate results
			resultVec[gThreadRowId] += mainVec[radiusOffsetRows + maskRowIn] * maskVec[maskRowIn];
		}
	}
}

void OneDConvolution::setContainer(const int& userInput)
{
	this->updateEventHandler(EventDirectives::populateContainer);

	// Users are displayed options 1 - 5 which translates to 0 - 4 for indexing
	int actualIndex{ userInput - 1 };

	// Prepare host containers
	this->processContainerSize(actualIndex);
	this->populateContainer(this->mOCHostInputVec, this->mOCHostMaskVec);
	this->setCurrSampleSize(actualIndex);

	// Prepare device containers
	this->prep1DKernelVars();
	this->updateMaskMemSize();
	this->allocateMemToDevice();
	this->copyHostToDevice();

	this->updateEventHandler(EventDirectives::populateContainerComplete);
}
void OneDConvolution::launchOp()
{
	this->updateEventHandler(EventDirectives::startOperation);
	this->OperationTimeHandler.resetStartTimer();

	// Launch Kernel on device
	oneConvKernel <<< this->mBLOCKS, this->mTHREADS >>> (this->mOCDeviceInputVec, this->mOCDeviceMaskVec, this->mOCDeviceOutputVec, this->mCurrSampleSize);

	this->OperationTimeHandler.collectElapsedTimeData();
	this->updateEventHandler(EventDirectives::endOperation);
}
void OneDConvolution::validateResults()
{
	this->updateEventHandler(EventDirectives::validateResults);
	this->copyDeviceToHost();
	this->freeDeviceData();

	// Assists in determining when convolution can occur to prevent out of bound errors
	// Used in conjunction with maskAttributes::maskOffset
	std::size_t radiusOffsetRows{ 0 };

	// Accumulates our results to check against resultVec
	std::size_t resultVar{0};

	// Determines result authenticity - Assigned false value when results don't match
	bool doesMatch{ true };

	for (auto rowIn{ 0 }; rowIn < this->mOCHostOutputVec.size() && doesMatch; ++rowIn)
	{
		// Reset resultVar to 0 on next element
		resultVar = 0;

		// Update offset value for that row
		radiusOffsetRows = rowIn - MaskAttributes::maskOffset;

		// For each mask row in mOCMaskVec
		for (auto maskRowIn{ 0 }; maskRowIn < MaskAttributes::maskDim; ++maskRowIn)
		{
			// Check if we're hanging off mask row
			if ((radiusOffsetRows + maskRowIn >= 0) && (radiusOffsetRows + maskRowIn < this->mOCHostOutputVec.size()))
			{
				// Accumulate results into resultVar
				resultVar += this->mOCHostInputVec[radiusOffsetRows + maskRowIn] * this->mOCHostMaskVec[maskRowIn];
			}
		}
		// Check accumulated resultVar value with corresponding value in resultVec
		if (resultVar != this->mOCHostOutputVec[rowIn])
			doesMatch = false;
	}

	this->setValidationStatus(doesMatch);

	// Assert and abort when results don't match
	assert(doesMatch && "Check failed! Accumulated resultVar value doesn't match corresponding value in mOCHostOutputVec (oneConv).");

	this->updateEventHandler(EventDirectives::resultsValidated);
}
void OneDConvolution::processContainerSize(const int& newIndex)
{
	// Convolution-specific mask vector size allocation - remains the same size regardless
	// If empty (first run), resize the mask vector - if already resized (second run +), ignore
	if (this->mOCHostMaskVec.empty())
		this->mOCHostMaskVec.resize(MaskAttributes::maskDim);

	if (this->isNewContainer() || this->isContainerSmallerSize(newIndex))
		this->resizeContainer(this->mSampleSizes[newIndex], this->mOCHostInputVec, this->mOCHostOutputVec);

	else if (this->isContainerSameSize(newIndex))
		return;

	else if (this->isContainerLargerSize(newIndex))
	{
		this->resizeContainer(this->mSampleSizes[newIndex], this->mOCHostInputVec, this->mOCHostOutputVec);
		// Non-binding - IDE will decide if this will execute
		this->shrinkContainer(this->mOCHostInputVec, this->mOCHostOutputVec);
	}

	// Only set next vecIndex if current container is smaller / larger / new
	this->setVecIndex(newIndex);
}

// CUDA Specific Functions
void OneDConvolution::allocateMemToDevice()
{
	cudaMalloc(&this->mOCDeviceInputVec, this->mMemSize);
	cudaMalloc(&this->mOCDeviceMaskVec, this->mMaskMemSize);
	cudaMalloc(&this->mOCDeviceOutputVec, this->mMemSize);
}
void OneDConvolution::copyHostToDevice()
{
	// Copy data from the host to the device using cudaMemcpy | .data() returns pointer to memory used by vector/array to store its owned elements
	cudaMemcpy(this->mOCDeviceInputVec, this->mOCHostInputVec.data(), this->mMemSize, cudaMemcpyHostToDevice);
	cudaMemcpy(this->mOCDeviceMaskVec, this->mOCHostMaskVec.data(), this->mMaskMemSize, cudaMemcpyHostToDevice);
}
void OneDConvolution::copyDeviceToHost()
{
	// Copy data from device back to host using cudaMemcpy
	cudaMemcpy(this->mOCHostOutputVec.data(), this->mOCDeviceOutputVec, this->mMemSize, cudaMemcpyDeviceToHost);
}
void OneDConvolution::freeDeviceData()
{
	cudaFree(this->mOCDeviceInputVec);
	cudaFree(this->mOCDeviceMaskVec);
	cudaFree(this->mOCDeviceOutputVec);
}