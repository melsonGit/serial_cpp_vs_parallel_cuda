#pragma once
#ifndef EVENT_HANDLER
#define EVENT_HANDLER

#include <string>
#include <cassert>

namespace OperationEvents
{
	enum class ContainerEvents
	{
		genericPopulationStart,
		genericPopulationComplete,
		maskPopulationStart,
		maskPopulationComplete,
		containerEventComplete,
	};
	enum class ArithmeticEvents
	{
		operationStart,
		arithmeticEventComplete,
	};
	enum class ValidationEvents
	{
		validationStart,
		validationFailedComplete,
		validationSuccessfulComplete,
		validationEventComplete,
	};
	enum class OutputToFileEvents
	{

	};
}

class OperationEventHandler
{
private:

	int mContainerEventTracker{ 0 };
	int mArithmeticEventTracker{ 0 };
	int mValidationEventTracker{ 0 };
	int mOutputToFileEventTracker{ 0 };

public:

	OperationEventHandler() = default;

	void trackEvents(const std::string& operation, const bool& hasMask);
	const bool containerPopulationEvent(const std::string& operation, const bool& hasMask);
	const bool arithmeticEvent(const std::string& operation);
	const bool validationEvent(const std::string& operation);
	const bool outputToFileEvent(const std::string& operation);
};
#endif