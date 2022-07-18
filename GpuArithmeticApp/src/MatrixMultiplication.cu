#include "../inc/MatrixMultiplication.cuh"

#include <algorithm>
#include <cassert>
#include <iostream>
#include <vector>

__global__ void matMultiKernel(const std::size_t* __restrict inputVecA, const std::size_t* __restrict inputVecB, std::size_t* __restrict resultVec,
	const std::size_t conSize)
{
	// Calculate and assign x / y dimensional thread a global thread ID
	std::size_t gThreadRowId = blockIdx.y * blockDim.y + threadIdx.y;
	std::size_t gThreadColId = blockIdx.x * blockDim.x + threadIdx.x;

	// Iterate over row, and down column
	resultVec[gThreadRowId * conSize + gThreadColId] = 0;

	// Above gThreadRow/ColId calculation skips traversal of row and col as it has already been calculated
	// This allows us to start straight at the rowColPairId
	for (auto rowColPairId{ 0 }; rowColPairId < conSize; ++rowColPairId)
	{
		// Accumulate results into resultVec
		resultVec[gThreadRowId * conSize + gThreadColId] += inputVecA[gThreadRowId * conSize + rowColPairId]
														  * inputVecB[rowColPairId * conSize + gThreadColId];
	}
}

// Reminder: When we switch from using native arrays to 2d vectors, Remove this
void MatrixMultiplication::tempConSizeInitTEMP()
{
	// Return values represent true native vector size i.e 1024^2 = 1048576
	// So our container size is really 1024

	bool badChoice{ false };

	switch (this->mMMHostOutputVec.size())
	{
	case 1048576: {this->tempConSize = 1024; return; break; }
	case 4194304: {this->tempConSize = 2048; return; break; }
	case 9437184: {this->tempConSize = 3072; return; break; }
	case 16777216: {this->tempConSize = 4096; return; break; }
	case 26214400: {this->tempConSize = 5120; return; break; }
	default: {badChoice = true; break; }
	}
	
	assert(!badChoice && "Bad tempConSizeInitTEMP() choice (matMulti).");
}
void MatrixMultiplication::setContainer(const int& userInput)
{
	this->updateEventHandler(EventDirectives::populateContainer);

	// Users are displayed options 1 - 5 which translates to 0 - 4 for indexing
	int actualIndex{ userInput - 1 };

	// Prepare host containers
	this->processContainerSize(actualIndex);
	this->populateContainer(this->mMMHostInputVecA, this->mMMHostInputVecB);
	this->setCurrSampleSize(actualIndex);

	// Prepare device containers
	this->tempConSizeInitTEMP();
	this->prep2DKernelVars();
	this->allocateMemToDevice();
	this->copyHostToDevice();
	this->updateDimStructs();
	
	this->updateEventHandler(EventDirectives::populateContainerComplete);
}
void MatrixMultiplication::launchOp()
{
	this->updateEventHandler(EventDirectives::startOperation);
	this->OperationTimeHandler.resetStartTimer();

	// Launch kernel on device
	matMultiKernel <<< this->mDimBlocks, this->mDimThreads >>> (this->mMMDeviceInputVecA, this->mMMDeviceInputVecB, 
		this->mMMDeviceOutputVec, this->tempConSize);

	this->OperationTimeHandler.collectElapsedTimeData();
	this->updateEventHandler(EventDirectives::endOperation);
}
void MatrixMultiplication::validateResults()
{
	this->updateEventHandler(EventDirectives::validateResults);
	this->copyDeviceToHost();
	this->freeDeviceData();

	// Accumulates our results to check against resultVec
	std::size_t resultVar{};

	// Determines result authenticity - Assigned false value when results don't match
	bool doesMatch{ true };

	// For each row
	for (auto rowIn{ 0 }; rowIn < this->tempConSize; ++rowIn)
	{
		for (auto colIn{ 0 }; colIn < this->tempConSize && doesMatch; ++colIn) // For each column in that row
		{
			// Reset resultVar to 0 on next element
			resultVar = 0;

			// For each row-column combination
			for (auto rowColPair{ 0 }; rowColPair < this->tempConSize; ++rowColPair)
			{
				// Accumulate results into resultVar
				resultVar += this->mMMHostInputVecA[rowIn * this->tempConSize + rowColPair] * this->mMMHostInputVecB[rowColPair * this->tempConSize + colIn];
			}
				
			// Check accumulated resultVar value with corresponding value in resultVec
			if (resultVar != this->mMMHostOutputVec[rowIn * this->tempConSize + colIn])
				doesMatch = false;
		}
	}

	this->setValidationStatus(doesMatch);

	// Assert and abort when results don't match
	assert(doesMatch && "Check failed! Accumulated resultVar value doesn't match corresponding value in mMMHostOutputVec (matMulti).");

	this->updateEventHandler(EventDirectives::resultsValidated);
}
void MatrixMultiplication::processContainerSize(const int& newIndex)
{
	if (this->isNewContainer() || this->isContainerSmallerSize(newIndex))
		this->resizeContainer(this->mSampleSizes[newIndex], this->mMMHostInputVecA, this->mMMHostInputVecB, this->mMMHostOutputVec);

	else if (this->isContainerSameSize(newIndex))
		return;

	else if (this->isContainerLargerSize(newIndex))
	{
		this->resizeContainer(this->mSampleSizes[newIndex], this->mMMHostInputVecA, this->mMMHostInputVecB, this->mMMHostOutputVec);
		// Non-binding - IDE will decide if this will execute
		this->shrinkContainer(this->mMMHostInputVecA, this->mMMHostInputVecB, this->mMMHostOutputVec);
	}
	
	// Only set next vecIndex if current container is smaller / larger / new
	this->setVecIndex(newIndex);
}

// CUDA Specific Functions
void MatrixMultiplication::allocateMemToDevice()
{
	cudaMalloc(&this->mMMDeviceInputVecA, this->mMemSize);
	cudaMalloc(&this->mMMDeviceInputVecB, this->mMemSize);
	cudaMalloc(&this->mMMDeviceOutputVec, this->mMemSize);
}
void MatrixMultiplication::copyHostToDevice()
{
	// Copy data from the host to the device using cudaMemcpy | .data() returns pointer to memory used by vector/array to store its owned elements
	cudaMemcpy(this->mMMDeviceInputVecA, this->mMMHostInputVecA.data(), this->mMemSize, cudaMemcpyHostToDevice);
	cudaMemcpy(this->mMMDeviceInputVecB, this->mMMHostInputVecB.data(), this->mMemSize, cudaMemcpyHostToDevice);
}
void MatrixMultiplication::copyDeviceToHost()
{
	// Copy data from device back to host using cudaMemcpy
	cudaMemcpy(this->mMMHostOutputVec.data(), this->mMMDeviceOutputVec, this->mMemSize, cudaMemcpyDeviceToHost);
}
void MatrixMultiplication::freeDeviceData()
{
	cudaFree(this->mMMDeviceInputVecA);
	cudaFree(this->mMMDeviceInputVecB);
	cudaFree(this->mMMDeviceOutputVec);
}