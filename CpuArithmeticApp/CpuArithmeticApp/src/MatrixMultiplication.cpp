#include "../inc/MatrixMultiplication.h"

#include <algorithm>
#include <cassert>
#include <iostream>
#include <vector>

void MatrixMultiplication::setContainer(const int& userInput)
{
	int actualIndex{ userInput - 1 };

	this->mMMInputVecA.resize(mSampleSizes[actualIndex], std::vector <std::size_t>(2, 0));
	this->mMMInputVecB.resize(mSampleSizes[actualIndex], std::vector <std::size_t>(2, 0));
	this->mMMOutputVec.resize(mSampleSizes[actualIndex], std::vector <std::size_t>(2, 0));

	populateContainer(this->mMMInputVecA, this->mMMInputVecB);
}

void MatrixMultiplication::launchOp()
{
    std::cout << "\nMatrix Multiplication: Populating complete.\n";
    std::cout << "\nMatrix Multiplication: Starting operation.\n";

    for (auto rowIn{ 0 }; rowIn < mMMOutputVec.size(); ++rowIn) // For each row
        for (auto colIn{ 0 }; colIn < mMMOutputVec[rowIn].size(); ++colIn) // For each column in that row
            for (auto rowColPair{ 0 }; rowColPair < mMMOutputVec[rowIn].size(); ++rowColPair) // For each row-column combination
                mMMOutputVec[rowIn][colIn] += mMMInputVecA[rowIn][rowColPair] * mMMInputVecB[rowColPair][colIn]; 

    std::cout << "\nMatrix Multiplication: Operation complete.\n";
}

void MatrixMultiplication::validateResults()
{
	std::cout << "\nMatrix Multiplication: Authenticating results.\n\n";

	// Accumulates our results to check against resultVec
	std::size_t resultVar{};

	// Determines result authenticity - Assigned false value when results don't match
	bool doesMatch{ true };

	// For each row
	for (auto rowIn{ 0 }; rowIn < mMMOutputVec.size(); ++rowIn) 
		for (auto colIn{ 0 }; colIn < mMMOutputVec[rowIn].size() && doesMatch; ++colIn) // For each column in that row
		{
			// Reset resultVar to 0 on next element
			resultVar = 0;

			// For each row-column combination
			for (auto rowColPair{ 0 }; rowColPair < mMMOutputVec[rowIn].size(); ++rowColPair)
				resultVar += mMMInputVecA[rowIn][rowColPair] * mMMInputVecB[rowColPair][colIn]; // Accumulate results into resultVar

			// Check accumulated resultVar value with corresponding value in resultVec
			if (resultVar != mMMOutputVec[rowIn][colIn])
				doesMatch = false;
		}

	// Assert and abort when results don't match
	assert(doesMatch && "Check failed! Accumulated resultVar value doesn't match corresponding value in resultVec (matMulti).");

	if (!doesMatch)
		std::cerr << "Matrix multiplication unsuccessful: output vector data does not match expected results.\n"
		<< "Timing results will be discarded.\n\n";
	else
		std::cout << "Matrix multiplication successful: output vector data matches expected results.\n"
		<< "Timing results will be recorded.\n\n";
}