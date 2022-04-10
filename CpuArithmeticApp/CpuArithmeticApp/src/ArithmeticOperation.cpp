#include "../inc/ArithmeticOperation.h"

#include <string_view>

const std::string_view ArithmeticOperation::getOperationName() const
{
	return this->mOperationName;
}

const std::size_t ArithmeticOperation::getOperationSampleSize(const int& option) const
{
	return this->mSampleSizes[option];
}