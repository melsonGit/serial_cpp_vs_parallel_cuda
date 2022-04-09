#include "../inc/ArithmeticOperation.h"

#include <iostream>
#include <string_view>

const std::string_view ArithmeticOperation::getOperationName() const
{
	return operationName;
}

const int& ArithmeticOperation::getOperationSampleSize(const int& option) const
{
	return this->sampleSizes[option];
}