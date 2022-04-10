#include "../inc/ArithmeticOperation.h"

#include <string_view>

const std::string_view ArithmeticOperation::getOpName() const
{
	return this->mOperationName;
}

const std::size_t ArithmeticOperation::getOpSampleSize(const int& option) const
{
	return this->mSampleSizes[option];
}

void ArithmeticOperation::startOpSeq(const int& userInput)
{
	setContainer(userInput);
	launchOp();
	validateResults();
}