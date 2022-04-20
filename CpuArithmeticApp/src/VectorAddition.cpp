#include "../inc/VectorAddition.h"

#include <algorithm>
#include <cassert>
#include <iostream>
#include <vector>

void VectorAddition::setContainer(const int& userInput) // another parameter pack for this?
{
	// Users are displayed options 1 - 5 which translates to 0 - 4 for indexing
	int actualIndex{ userInput - 1 };
	// First run check - any number outside 0 - 6 is fine but just to be safe
	constexpr int firstRun{ 99 }; 

	// If first run - we'll re-size regardless
	if (this->getCurrentVecSize() == firstRun)
	{
		this->setCurrentVecSize(actualIndex);
		this->mVAInputVecA.resize(mSampleSizes[actualIndex]);
		this->mVAInputVecB.resize(mSampleSizes[actualIndex]);
		this->mVAOutputVec.resize(mSampleSizes[actualIndex]);
	}
	else if (actualIndex == this->getCurrentVecSize())
	{
		// or we jump straight to populating if user selected same sample size as last run - don't resize, just re-populate vectors
		populateContainer(this->mVAInputVecA, this->mVAInputVecB);
	}
	else if (actualIndex < this->getCurrentVecSize()) // If current sample selection is lower than previous run - resize() and then shrink_to_fit().
	{
		this->setCurrentVecSize(actualIndex);
		this->mVAInputVecA.resize(mSampleSizes[actualIndex]);
		this->mVAInputVecB.resize(mSampleSizes[actualIndex]);
		this->mVAOutputVec.resize(mSampleSizes[actualIndex]);
		// Non-binding - IDE will decide if this will execute
		this->mVAInputVecA.shrink_to_fit();
		this->mVAInputVecB.shrink_to_fit();
		this->mVAOutputVec.shrink_to_fit();
	}
	else // If selection is higher than last run
	{
		this->setCurrentVecSize(actualIndex);
		this->mVAInputVecA.resize(mSampleSizes[actualIndex]);
		this->mVAInputVecB.resize(mSampleSizes[actualIndex]);
		this->mVAOutputVec.resize(mSampleSizes[actualIndex]);
	}

	populateContainer(this->mVAInputVecA, this->mVAInputVecB);
}
void VectorAddition::launchOp()
{
	// Reminder: These event outputs could be placed into a class somewhere.......
	std::cout << "\nVector Addition: Populating complete.\n";
	std::cout << "\nVector Addition: Starting operation.\n";

	// Add contents from inputVecA and inputVecB into resultVec
	transform(this->mVAInputVecA.begin(), this->mVAInputVecA.end(), this->mVAInputVecB.begin(), this->mVAOutputVec.begin(),
		[](auto a, auto b) {return a + b; });

	std::cout << "\nVector Addition: Operation complete.\n";
}
void VectorAddition::validateResults() 
{
	std::cout << "\nVector Addition: Authenticating results.\n\n";

	// Determines result authenticity - Assigned false value when results don't match
	bool doesMatch{ true };

	// For each row in inputVecA/B 
	for (auto rowId{ 0 }; rowId < this->mVAOutputVec.size() && doesMatch; ++rowId)
	{
		// Check addition of both rows matches value in corresponding row in resultVec
		if ((this->mVAInputVecA[rowId] + this->mVAInputVecB[rowId]) != this->mVAOutputVec[rowId])
			doesMatch = false;
	}
	// Assert and abort when results don't match
	assert(doesMatch && "Check failed! Addition of mVAInputVecA / mVAInputVecB values don't match corresponding values in mVAOutputVec (vecAdd).");

	if (!doesMatch)
		std::cerr << "Vector Addition unsuccessful: output vector data does not match expected results.\n"
		<< "Timing results will be discarded.\n\n";
	else
		std::cout << "Vector Addition successful: output vector data matches expected results.\n"
		<< "Timing results will be recorded.\n\n";
}