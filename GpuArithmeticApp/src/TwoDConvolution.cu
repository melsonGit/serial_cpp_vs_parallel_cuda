#include "../inc/TwoDConvolution.cuh"
#include "../inc/MaskAttributes.h"

#include <cassert>
#include <iostream>
#include <vector>

__global__ void twoConvKernel(const std::size_t* __restrict mainVec, const std::size_t* __restrict maskVec, std::size_t* __restrict resultVec, const std::size_t conSize)
{
	// Calculate and assign x / y dimensional thread a global thread ID
	std::size_t gThreadRowId = blockIdx.y * blockDim.y + threadIdx.y;
	std::size_t gThreadColId = blockIdx.x * blockDim.x + threadIdx.x;

	// Temp values to work around device code issue
	const std::size_t maskDim{ 7 };
	const std::size_t maskOffset{ maskDim / 2 };

	// Radius rows/cols will determine when convolution occurs to prevent out of bound errors
	// twoConv utilises one for rows AND columns as we're dealing with a 2D mask vector
	std::size_t radiusOffsetRows{ gThreadRowId - maskOffset };
	std::size_t radiusOffsetCols{ gThreadColId - maskOffset };

	// Accumulate results - temp?
	std::size_t resultVar{};

	// For each row
	for (auto rowId{ 0 }; rowId < maskDim; ++rowId)
	{
		// For each column in that row
		for (auto colId{ 0 }; colId < maskDim; ++colId)
		{
			// Range check for rows
			if ((radiusOffsetRows + rowId) >= 0 && (radiusOffsetRows + rowId) < conSize)
			{
				// Range check for columns
				if ((radiusOffsetCols + colId) >= 0 && (radiusOffsetCols + colId) < conSize)
				{
					// Accumulate results into resultVar
					resultVec[gThreadRowId * conSize + gThreadColId] += mainVec[(radiusOffsetRows + rowId) * conSize + (radiusOffsetCols + colId)] * maskVec[rowId * maskDim + colId];
				}
			}
		}
	}
	// Assign resultVec the accumulated value of resultVar
	//resultVec[gThreadRowId * conSize + gThreadColId] = resultVar;
}

// Reminder: When we switch from using native arrays to 2d vectors, Remove this
void TwoDConvolution::tempConSizeInitTEMP()
{
	// Return values represent true native vector size i.e 4096^2 = 16777216
	// So our container size is really 4096

	bool badChoice{ false };

	switch (this->mTCHostOutputVec.size())
	{
	case 16777216: {this->tempConSize = 4096; return; break; }
	case 26214400: {this->tempConSize = 5120; return; break; }
	case 37748736: {this->tempConSize = 6144; return; break; }
	case 67108864: {this->tempConSize = 8192; return; break; }
	case 104857600: {this->tempConSize = 10240; return; break; }
	default: {badChoice = true; break; }
	}

	assert(!badChoice && "Bad tempConSizeInitTEMP() choice (twoConv).");
}
void TwoDConvolution::setContainer(const int& userInput)
{
	this->updateEventHandler(EventDirectives::populateContainer);

	// Users are displayed options 1 - 5 which translates to 0 - 4 for indexing
	int actualIndex{ userInput - 1 };

	// Prepare host containers
	this->processContainerSize(actualIndex);
	this->populateContainer(this->mTCHostInputVec, this->mTCHostMaskVec);
	this->setCurrSampleSize(actualIndex);

	// Prepare device containers
	this->tempConSizeInitTEMP();
	this->prep2DKernelVars();
	this->update2DMaskMemSize();
	this->allocateMemToDevice();
	this->updateDimStructs();
	this->copyHostToDevice();

	this->updateEventHandler(EventDirectives::populateContainerComplete);
}
void TwoDConvolution::launchOp()
{
	this->updateEventHandler(EventDirectives::startOperation);
	this->OperationTimeHandler.resetStartTimer();

	// Launch Kernel on device
	twoConvKernel <<< this->mDimBlocks, this->mDimThreads >>> (this->mTCDeviceInputVec, this->mTCDeviceMaskVec, this->mTCDeviceOutputVec, this->tempConSize);

	this->OperationTimeHandler.collectElapsedTimeData();
	this->updateEventHandler(EventDirectives::endOperation);
}
void TwoDConvolution::validateResults()
{
	this->updateEventHandler(EventDirectives::validateResults);
	this->copyDeviceToHost();
	this->freeDeviceData();

	// Assists in determining when convolution can occur to prevent out of bound errors
	// Used in conjunction with maskAttributes::maskOffset
	std::size_t radiusOffsetRows{ 0 };
	std::size_t radiusOffsetCols{ 0 };

	// Replace this var with mTCOutputVec.size() in all if loop conditional statements when we use 2d vectors
	// this->tempConSize

	// Accumulates our results
	std::size_t resultVar{};

	// Determines result authenticity - Assigned false value when results don't match
	bool doesMatch{ true };

	// For each row in mainVec
	for (auto rowIn{ 0 }; rowIn < this->tempConSize; ++rowIn)
	{
		// For each column in that row
		for (auto colIn{ 0 }; colIn < this->tempConSize && doesMatch; ++colIn)
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
					if (radiusOffsetRows >= 0 && radiusOffsetRows < this->tempConSize)
					{
						// Check if we're hanging off mask column
						if (radiusOffsetCols >= 0 && radiusOffsetCols < this->tempConSize)
						{
							// Accumulate results into resultVar
							resultVar += this->mTCHostInputVec[radiusOffsetRows * this->tempConSize + radiusOffsetCols] * this->mTCHostMaskVec[maskRowIn * MaskAttributes::maskDim + maskColIn];
						}
					}
				}
			}

			// Check accumulated resultVar value with corresponding value in resultVec
			if (resultVar != this->mTCHostOutputVec[rowIn * this->tempConSize + colIn])
				doesMatch = false;
		}
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
	if (this->mTCHostMaskVec.empty())
		this->mTCHostMaskVec.resize(MaskAttributes::maskDim * MaskAttributes::maskDim);

	if (this->isNewContainer() || this->isContainerSmallerSize(newIndex))
		this->resizeContainer(this->mSampleSizes[newIndex], this->mTCHostInputVec, this->mTCHostOutputVec);

	else if (this->isContainerSameSize(newIndex))
		return;

	else if (this->isContainerLargerSize(newIndex))
	{
		this->resizeContainer(this->mSampleSizes[newIndex], this->mTCHostInputVec, this->mTCHostOutputVec);
		// Non-binding - IDE will decide if this will execute
		this->shrinkContainer(this->mTCHostInputVec, this->mTCHostOutputVec);
	}

	// Only set next vecIndex if current container is smaller / larger / new
	this->setVecIndex(newIndex);
}

// CUDA Specific Functions
void TwoDConvolution::allocateMemToDevice()
{
	cudaMalloc(&this->mTCDeviceInputVec, this->mMemSize);
	cudaMalloc(&this->mTCDeviceMaskVec, this->m2DMaskMemSize);
	cudaMalloc(&this->mTCDeviceOutputVec, this->mMemSize);
}
void TwoDConvolution::copyHostToDevice()
{
	// Copy data from the host to the device using cudaMemcpy | .data() returns pointer to memory used by vector/array to store its owned elements
	cudaMemcpy(this->mTCDeviceInputVec, this->mTCHostInputVec.data(), this->mMemSize, cudaMemcpyHostToDevice);
	cudaMemcpy(this->mTCDeviceMaskVec, this->mTCHostMaskVec.data(), this->m2DMaskMemSize, cudaMemcpyHostToDevice);
}
void TwoDConvolution::copyDeviceToHost()
{
	// Copy data from device back to host using cudaMemcpy
	cudaMemcpy(this->mTCHostOutputVec.data(), this->mTCDeviceOutputVec, this->mMemSize, cudaMemcpyDeviceToHost);
}
void TwoDConvolution::freeDeviceData()
{
	cudaFree(this->mTCDeviceInputVec);
	cudaFree(this->mTCDeviceMaskVec);
	cudaFree(this->mTCDeviceOutputVec);
}