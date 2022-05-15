#include "../inc/MatrixMultiplication.h"

#include <algorithm>
#include <cassert>
#include <iostream>
#include <vector>

void MatrixMultiplication::setContainer(const int& userInput)
{
	this->updateEventHandler(EventDirectives::populateContainer);

	// Users are displayed options 1 - 5 which translates to 0 - 4 for indexing
	int actualIndex{ userInput - 1 };

	this->processContainerSize(actualIndex);

	this->populateContainer(this->mMMInputVecA, this->mMMInputVecB);

	this->setCurrSampleSize(actualIndex);

	this->updateEventHandler(EventDirectives::populateContainerComplete);
}
void MatrixMultiplication::launchOp()
{
	this->updateEventHandler(EventDirectives::startOperation);
	this->OperationTimeHandler.resetStartTimer();

    for (auto rowIn{ 0 }; rowIn < this->mMMOutputVec.size(); ++rowIn) // For each row
		for (auto colIn{ 0 }; colIn < this->mMMOutputVec[rowIn].size(); ++colIn) // For each column in that row
		{
			// Reset next mMMOutputVec[rowIn][colIn] to 0 on next element
			this->mMMOutputVec[rowIn][colIn] = 0;

			for (auto rowColPair{ 0 }; rowColPair < this->mMMOutputVec[rowIn].size(); ++rowColPair)// For each row-column combination
				this->mMMOutputVec[rowIn][colIn] += this->mMMInputVecA[rowIn][rowColPair] * this->mMMInputVecB[rowColPair][colIn];
		}

	this->OperationTimeHandler.collectElapsedTimeData();
	this->updateEventHandler(EventDirectives::endOperation);
}
void MatrixMultiplication::validateResults()
{
	this->updateEventHandler(EventDirectives::validateResults);

	// Accumulates our results to check against resultVec
	std::size_t resultVar{};

	// Determines result authenticity - Assigned false value when results don't match
	bool doesMatch{ true };

	// For each row
	for (auto rowIn{ 0 }; rowIn < this->mMMOutputVec.size(); ++rowIn)
	{
		for (auto colIn{ 0 }; colIn < this->mMMOutputVec[rowIn].size() && doesMatch; ++colIn) // For each column in that row
		{
			// Reset resultVar to 0 on next element
			resultVar = 0;

			// For each row-column combination
			for (auto rowColPair{ 0 }; rowColPair < this->mMMOutputVec[rowIn].size(); ++rowColPair)
			{
				resultVar += this->mMMInputVecA[rowIn][rowColPair] * this->mMMInputVecB[rowColPair][colIn]; // Accumulate results into resultVar
			}
				
			// Check accumulated resultVar value with corresponding value in resultVec
			if (resultVar != this->mMMOutputVec[rowIn][colIn])
				doesMatch = false;
		}
	}

	this->setValidationStatus(doesMatch);

	// Assert and abort when results don't match
	assert(doesMatch && "Check failed! Accumulated resultVar value doesn't match corresponding value in mMMOutputVec (matMulti).");

	this->updateEventHandler(EventDirectives::resultsValidated);
}
void MatrixMultiplication::processContainerSize(const int& newIndex)
{
	if (this->isNewContainer() || this->isContainerSmallerSize(newIndex))
		this->resizeContainer(this->mSampleSizes[newIndex], this->mMMInputVecA, this->mMMInputVecB, this->mMMOutputVec);

	else if (this->isContainerSameSize(newIndex))
		return;

	else if (this->isContainerLargerSize(newIndex))
	{
		this->resizeContainer(this->mSampleSizes[newIndex], this->mMMInputVecA, this->mMMInputVecB, this->mMMOutputVec);
		// Non-binding - IDE will decide if this will execute
		this->shrinkContainer(this->mMMInputVecA, this->mMMInputVecB, this->mMMOutputVec);
	}
	
	// Only set next vecIndex if current container is smaller / larger / new
	this->setVecIndex(newIndex);
}