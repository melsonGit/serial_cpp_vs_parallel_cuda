#include "../inc/ArithmeticOperation.h"

#include <string_view>

// Getters

const int ArithmeticOperation::getCurrentVecSize() const
{
	return this->currentVecSize;
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

void ArithmeticOperation::setCurrentVecSize(const int& newSize)
{
	this->currentVecSize = newSize;
}
void ArithmeticOperation::startOpSeq(const int& userInput)
{
	this->setContainer(userInput);
	this->launchOp();
	this->validateResults();
}