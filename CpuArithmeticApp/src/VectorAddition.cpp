#include "../inc/VectorAddition.h"

#include <algorithm>
#include <cassert>
#include <iostream>
#include <vector>

void VectorAddition::setContainer(const int& userInput)
{
	// Users are displayed options 1 - 5 which translates to 0 - 4 for indexing
	int actualIndex{ userInput - 1 };
	// First run check - any number outside 0 - 6 is fine but just to be safe
	constexpr int firstRun{ 99 }; 

	if (this->getVecIndex() == firstRun)
	{
		// If first run - we'll re-size regardless
		this->setVecIndex(actualIndex);
		this->resizeContainer(this->mSampleSizes[actualIndex], this->mVAInputVecA, this->mVAInputVecB, this->mVAOutputVec);
	}
	else if (actualIndex < this->getVecIndex())
	{
		// If current sample selection is lower than previous run - resize() and then shrink_to_fit().
		this->setVecIndex(actualIndex);
		this->resizeContainer(this->mSampleSizes[actualIndex], this->mVAInputVecA, this->mVAInputVecB, this->mVAOutputVec);
		// Non-binding - IDE will decide if this will execute
		this->shrinkContainer(mVAInputVecA, mVAInputVecB, mVAOutputVec);
	}
	else if (actualIndex > this->getVecIndex())
	{
		// If selection is higher than last run
		this->setVecIndex(actualIndex);
		this->resizeContainer(this->mSampleSizes[actualIndex], this->mVAInputVecA, this->mVAInputVecB, this->mVAOutputVec);
	}

	this->updateEventHandler(EventDirectives::populateContainer);

	// or we jump straight to populating if user selected same sample size as last run - don't resize, just re-populate vectors
	this->populateContainer(this->mVAInputVecA, this->mVAInputVecB);

	this->setCurrSampleSize(actualIndex);

	this->updateEventHandler(EventDirectives::populateContainerComplete);
}
void VectorAddition::launchOp()
{
	this->updateEventHandler(EventDirectives::startOperation);
	this->OperationTimer.resetStartTimer();

	// Add contents from inputVecA and inputVecB into resultVec
	transform(this->mVAInputVecA.begin(), this->mVAInputVecA.end(), this->mVAInputVecB.begin(), this->mVAOutputVec.begin(),
		[](auto a, auto b) {return a + b; });

	this->OperationTimer.collectElapsedTimeData();
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