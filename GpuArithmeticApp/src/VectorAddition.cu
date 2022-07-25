#include "../inc/VectorAddition.cuh"

#include <algorithm>
#include <cassert>
#include <iostream>
#include <vector>

// Cannot be a member function
__global__ void vecAddKernel(const std::size_t* __restrict inputVecA, const std::size_t* __restrict inputVecB, 
	std::size_t* __restrict resultVec, const std::size_t conSize)
{
	// Calculate and assign x dimensional thread a global thread ID
	std::size_t gThreadRowId = blockIdx.x * blockDim.x + threadIdx.x;

	// If the threads gThreadRowId is below sample size
	// Add contents from inputVecA and inputVecB into resultVec 
	if (gThreadRowId < conSize)
		resultVec[gThreadRowId] = inputVecA[gThreadRowId] + inputVecB[gThreadRowId];
}

void VectorAddition::setContainer(const int& userInput)
{
	this->updateEventHandler(EventDirectives::populateContainer);

	// Users are displayed options 1 - 5 which translates to 0 - 4 for indexing
	int actualIndex{ userInput - 1 };

	// Prepare host containers
	this->processContainerSize(actualIndex);
	this->populateContainer(this->mVAHostInputVecA, this->mVAHostInputVecB);
	this->setCurrSampleSize(actualIndex);

	// Prepare device containers
	this->update1DKernelVars();
	this->allocateMemToDevice();
	this->copyHostToDevice();

	this->updateEventHandler(EventDirectives::populateContainerComplete);
}
void VectorAddition::launchOp()
{
	this->updateEventHandler(EventDirectives::startOperation);
	this->OperationTimeHandler.resetStartTimer();

	// Launch Kernel on device
	vecAddKernel <<< this->mBLOCKS, this->mTHREADS >>> (this->mVADeviceInputVecA, this->mVADeviceInputVecB, this->mVADeviceOutputVec, this->mCurrSampleSize);

	this->OperationTimeHandler.collectElapsedTimeData();
	this->updateEventHandler(EventDirectives::endOperation);
}
void VectorAddition::validateResults() 
{
	this->updateEventHandler(EventDirectives::validateResults);
	this->copyDeviceToHost();
	this->freeDeviceData();

	// Determines result authenticity - Assigned false value when results don't match
	bool doesMatch{ true };

	// For each row in inputVecA/B 
	for (auto rowId{ 0 }; rowId < this->mVAHostOutputVec.size() && doesMatch; ++rowId)
	{
		// Check addition of both rows matches value in corresponding row in resultVec
		if ((this->mVAHostInputVecA[rowId] + this->mVAHostInputVecB[rowId]) != this->mVAHostOutputVec[rowId])
			doesMatch = false;
	}

	this->setValidationStatus(doesMatch);

	// Assert and abort when results don't match
	assert(doesMatch && "Check failed! Addition of mVAHostInputVecA / mVAHostInputVecB values don't match corresponding values in mVAHostOutputVec (vecAdd).");

	this->updateEventHandler(EventDirectives::resultsValidated);
}
void VectorAddition::processContainerSize(const int& newIndex)
{
	if (this->isNewContainer() || this->isContainerSmallerSize(newIndex))
		this->resizeContainer(this->mSampleSizes[newIndex], this->mVAHostInputVecA, this->mVAHostInputVecB, this->mVAHostOutputVec);

	else if (this->isContainerSameSize(newIndex))
		return;

	else if (this->isContainerLargerSize(newIndex))
	{
		this->resizeContainer(this->mSampleSizes[newIndex], this->mVAHostInputVecA, this->mVAHostInputVecB, this->mVAHostOutputVec);
		// Non-binding - IDE will decide if this will execute
		this->shrinkContainer(mVAHostInputVecA, mVAHostInputVecB, mVAHostOutputVec);
	}

	// Only set next vecIndex if current container is smaller / larger / new
	this->setVecIndex(newIndex);
}

// CUDA Specific Functions
void VectorAddition::allocateMemToDevice()
{
	cudaMalloc(&this->mVADeviceInputVecA, this->mMemSize);
	cudaMalloc(&this->mVADeviceInputVecB, this->mMemSize);
	cudaMalloc(&this->mVADeviceOutputVec, this->mMemSize);
}
void VectorAddition::copyHostToDevice()
{
	// Copy data from the host to the device using cudaMemcpy | .data() returns pointer to memory used by vector/array to store its owned elements
	cudaMemcpy(this->mVADeviceInputVecA, this->mVAHostInputVecA.data(), this->mMemSize, cudaMemcpyHostToDevice);
	cudaMemcpy(this->mVADeviceInputVecB, this->mVAHostInputVecB.data(), this->mMemSize, cudaMemcpyHostToDevice);
}
void VectorAddition::copyDeviceToHost()
{
	// Copy data from device back to host using cudaMemcpy
	cudaMemcpy(this->mVAHostOutputVec.data(), this->mVADeviceOutputVec, this->mMemSize, cudaMemcpyDeviceToHost);
}
void VectorAddition::freeDeviceData()
{
	cudaFree(this->mVADeviceInputVecA);
	cudaFree(this->mVADeviceInputVecB);
	cudaFree(this->mVADeviceOutputVec);
}