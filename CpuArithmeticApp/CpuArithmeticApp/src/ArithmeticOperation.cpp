#include "../inc/ArithmeticOperation.h"

#include <iostream>
#include <string_view>

const std::string_view ArithmeticOperation::getOperationName() const
{
	return mOperationName;
}

const int& ArithmeticOperation::getOperationSampleSize(const int& option) const
{
	switch (option)
	{
	case 1:
		return this->mSampleSizes[0];
	case 2:
		return this->mSampleSizes[1];
	case 3:
		return this->mSampleSizes[2];
	case 4:
		return this->mSampleSizes[3];
	case 5:
		return this->mSampleSizes[4];
	default:
	{
		std::cout << "Invalid option!\n";
		break;
	}
	}
}