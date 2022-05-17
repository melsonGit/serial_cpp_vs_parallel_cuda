#include "../inc/VectorAddition.h"

#include <algorithm>
#include <cassert>
#include <iostream>
#include <vector>

void VectorAddition::setContainer(const int& userInput)
{
	this->updateEventHandler(EventDirectives::populateContainer);

	// Users are displayed options 1 - 5 which translates to 0 - 4 for indexing
	int actualIndex{ userInput - 1 };

	this->processContainerSize(actualIndex);

	this->populateContainer(this->mVAInputVecA, this->mVAInputVecB);

	this->setCurrSampleSize(actualIndex);

	this->updateEventHandler(EventDirectives::populateContainerComplete);
}
void VectorAddition::launchOp()
{
	this->updateEventHandler(EventDirectives::startOperation);
	this->OperationTimeHandler.resetStartTimer();

	// Add contents from inputVecA and inputVecB into resultVec
	transform(this->mVAInputVecA.begin(), this->mVAInputVecA.end(), this->mVAInputVecB.begin(), this->mVAOutputVec.begin(),
		[](auto a, auto b) {return a + b; });

	this->OperationTimeHandler.collectElapsedTimeData();
	this->updateEventHandler(EventDirectives::endOperation);
}
void VectorAddition::validateResults() 
{
	this->updateEventHandler(EventDirectives::validateResults);

	// Determines result authenticity - Assigned false value when results don't match
	bool doesMatch{ true };

	// For each row in inputVecA/B 
	for (auto rowId{ 0 }; rowId < this->mVAOutputVec.size() && doesMatch; ++rowId)
	{
		// Check addition of both rows matches value in corresponding row in resultVec
		if ((this->mVAInputVecA[rowId] + this->mVAInputVecB[rowId]) != this->mVAOutputVec[rowId])
			doesMatch = false;
	}

	this->setValidationStatus(doesMatch);

	// Assert and abort when results don't match
	assert(doesMatch && "Check failed! Addition of mVAInputVecA / mVAInputVecB values don't match corresponding values in mVAOutputVec (vecAdd).");

	this->updateEventHandler(EventDirectives::resultsValidated);
}
void VectorAddition::processContainerSize(const int& newIndex)
{
	if (this->isNewContainer() || this->isContainerSmallerSize(newIndex))
		this->resizeContainer(this->mSampleSizes[newIndex], this->mVAInputVecA, this->mVAInputVecB, this->mVAOutputVec);

	else if (this->isContainerSameSize(newIndex))
		return;

	else if (this->isContainerLargerSize(newIndex))
	{
		this->resizeContainer(this->mSampleSizes[newIndex], this->mVAInputVecA, this->mVAInputVecB, this->mVAOutputVec);
		// Non-binding - IDE will decide if this will execute
		this->shrinkContainer(mVAInputVecA, mVAInputVecB, mVAOutputVec);
	}

	// Only set next vecIndex if current container is smaller / larger / new
	this->setVecIndex(newIndex);
}