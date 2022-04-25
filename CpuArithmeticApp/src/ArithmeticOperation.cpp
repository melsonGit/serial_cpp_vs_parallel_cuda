#include "../inc/ArithmeticOperation.h"

//void ArithmeticOperation::recordResults()
//{
//	this->
//}

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