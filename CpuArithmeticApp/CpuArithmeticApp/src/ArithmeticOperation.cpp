#include "../inc/ArithmeticOperation.h"

#include <iostream>
#include <string_view>

const std::string_view ArithmeticOperation::getOperationName() const
{
	return this->mOperationName;
}

const int& ArithmeticOperation::getOperationSampleSize(const int& option) const
{
	return this->mSampleSizes[option];
}