#include "../inc/ArithmeticOperation.h"

void ArithmeticOperation::recordResults()
{
	/*
		Uses name, sample size and timing results from OperationTimer
		Locate a folder and .csv file to output to
		check if one exists or not
		create one if needed or enter into existing file
		record results, grouped by operation then grouped by sample size
	*/
}
void ArithmeticOperation::bestOperationTimes()
{
	/*
		Uses name, sample size and timing results from OperationTimer
		Caches times into a hashmap (even when the program is closed), looks for results stored in files used by recordResults(),
		and checks files for each operations fastest time in each sample size range
	*/
}

// Getters

const bool ArithmeticOperation::getValidationStatus() const
{
	return this->hasPassedValidation;
}
const int ArithmeticOperation::getVecIndex() const
{
	return this->vecIndex;
}
const std::size_t ArithmeticOperation::getCurrSampleSize() const
{
	return this->currSampleSize;
}
const bool ArithmeticOperation::getMaskStatus() const
{
	return this->hasMask;
}
const std::string ArithmeticOperation::getOpName() const
{
	return this->mOperationName;
}
const std::size_t ArithmeticOperation::getOpSampleSize(const int& option) const
{
	return this->mSampleSizes[option];
}

// Setters
void ArithmeticOperation::setCurrSampleSize(const int& index)
{
	this->currSampleSize = this->mSampleSizes[index];
}
void ArithmeticOperation::setValidationStatus(const bool& validationResults)
{
	this->hasPassedValidation = validationResults;
}
void ArithmeticOperation::setVecIndex(const int& newIndex)
{
	this->vecIndex = newIndex;
}
void ArithmeticOperation::startOpSeq(const int& userInput)
{
	this->setContainer(userInput);
	this->launchOp();
	this->validateResults();
}