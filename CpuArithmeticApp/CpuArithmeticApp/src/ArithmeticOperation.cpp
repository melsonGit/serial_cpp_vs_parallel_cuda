#include "../inc/ArithmeticOperation.h"

#include <iostream>
#include <string_view>

const std::string_view ArithmeticOperation::getOperationName() const
{
	return operationName;
}

const int& ArithmeticOperation::getOperationSampleSize(const int& option) const
{
	switch (option)
	{
	case 1:
		return this->sampleSizes[0];
	case 2:
		return this->sampleSizes[1];
	case 3:
		return this->sampleSizes[2];
	case 4:
		return this->sampleSizes[3];
	case 5:
		return this->sampleSizes[4];
	default:
	{
		std::cout << "Invalid option!\n";
		break;
	}
	}
}