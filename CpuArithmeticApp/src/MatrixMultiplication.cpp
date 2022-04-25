#include "../inc/MatrixMultiplication.h"

#include <algorithm>
#include <cassert>
#include <iostream>
#include <vector>

void MatrixMultiplication::setContainer(const int& userInput)
{
	this->OperationEventHandler.processEvent();

	// Users are displayed options 1 - 5 which translates to 0 - 4 for indexing
	int actualIndex{ userInput - 1 };
	// First run check - any number outside 0 - 6 is fine but just to be safe
	constexpr int firstRun{ 99 };

	if (this->getVecIndex() == firstRun)
	{
		// If first run - we'll re-size regardless
		this->setVecIndex(actualIndex);
		this->mMMInputVecA.resize(this->mSampleSizes[actualIndex], std::vector <std::size_t>(2, 0));
		this->mMMInputVecB.resize(this->mSampleSizes[actualIndex], std::vector <std::size_t>(2, 0));
		this->mMMOutputVec.resize(this->mSampleSizes[actualIndex], std::vector <std::size_t>(2, 0));
	}
	else if (actualIndex < this->getVecIndex()) 
	{
		// If current sample selection is lower than previous run - resize() and then shrink_to_fit().
		this->setVecIndex(actualIndex);

		this->mMMInputVecA.resize(this->mSampleSizes[actualIndex], std::vector <std::size_t>(2, 0));
		this->mMMInputVecB.resize(this->mSampleSizes[actualIndex], std::vector <std::size_t>(2, 0));
		this->mMMOutputVec.resize(this->mSampleSizes[actualIndex], std::vector <std::size_t>(2, 0));
		// Non-binding - IDE will decide if this will execute
		this->mMMInputVecA.shrink_to_fit();
		this->mMMInputVecB.shrink_to_fit();
		this->mMMOutputVec.shrink_to_fit();
	}
	else if (actualIndex > this->getVecIndex())
	{
		// If selection is higher than last run
		this->setVecIndex(actualIndex);
		this->mMMInputVecA.resize(this->mSampleSizes[actualIndex], std::vector <std::size_t>(2, 0));
		this->mMMInputVecB.resize(this->mSampleSizes[actualIndex], std::vector <std::size_t>(2, 0));
		this->mMMOutputVec.resize(this->mSampleSizes[actualIndex], std::vector <std::size_t>(2, 0));
	}

	// or we jump straight to populating if user selected same sample size as last run - don't resize, just re-populate vectors
	this->populateContainer(this->mMMInputVecA, this->mMMInputVecB);

	this->setCurrSampleSize(actualIndex);

	this->OperationEventHandler.processEvent();
	this->OperationEventHandler.processEvent();
}
void MatrixMultiplication::launchOp()
{
	this->OperationEventHandler.processEvent();
	this->OperationTimer.resetStartTimer();

    for (auto rowIn{ 0 }; rowIn < this->mMMOutputVec.size(); ++rowIn) // For each row
		for (auto colIn{ 0 }; colIn < this->mMMOutputVec[rowIn].size(); ++colIn) // For each column in that row
		{
			// Reset next mMMOutputVec[rowIn][colIn] to 0 on next element
			this->mMMOutputVec[rowIn][colIn] = 0;

			for (auto rowColPair{ 0 }; rowColPair < this->mMMOutputVec[rowIn].size(); ++rowColPair)// For each row-column combination
				this->mMMOutputVec[rowIn][colIn] += this->mMMInputVecA[rowIn][rowColPair] * this->mMMInputVecB[rowColPair][colIn];
		}

	this->OperationTimer.collectElapsedTimeData();
	this->OperationEventHandler.processEvent();
}
void MatrixMultiplication::validateResults()
{
	this->OperationEventHandler.processEvent();

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

	this->OperationEventHandler.processEvent();
}