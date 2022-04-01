#include "../inc/ArithmeticOperation.h"

#include <string_view>

const std::string_view ArithmeticOperation::getOperationName() const
{
	return operationName;
}