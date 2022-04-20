#include "../inc/OperationEventHandler.h"

#include <iostream>
#include <string>
#include <cassert>

using namespace OperationEvents;

/*
	=====****| Parameter Legend |****=====

	operation: Denotes the operation we're processing throughout events
	event:     Denotes the specific event within an event function
	hasMask:   Denotes if the operation uses a mask vector in ther arithmetic operation

*/

void OperationEventHandler::trackEvents(const std::string& operation, const bool& hasMask)
{
	std::string j = "true";
	int i = 1;
	bool h = true;

	while (!containerPopulationEvent(j, i));
	while (!arithmeticEvent(j));
	while (!validationEvent(j));
}
const bool OperationEventHandler::containerPopulationEvent(const std::string& operation, const bool& hasMask)
{
	using enum ContainerEvents;
	int skipEvent{ 3 };
	bool eventComplete{ false };

	switch (static_cast<ContainerEvents>(this->mContainerEventTracker))
	{
	case genericPopulationStart:
	{
		std::cout << '\n' << operation << ": Populating main containers.\n";
		mContainerEventTracker++;
		break;
	}
	case genericPopulationComplete:
	{
		std::cout << '\n' << operation << ": Main container population complete.\n";

		if (hasMask)
			mContainerEventTracker++;
		else
			mContainerEventTracker += skipEvent;
		break;
	}
	case maskPopulationStart:
	{
		std::cout << '\n' << operation << ": Populating mask containers.\n";
		mContainerEventTracker++;
		break;
	}
	case maskPopulationComplete:
	{
		std::cout << '\n' << operation << ": Mask container population complete.\n";
		mContainerEventTracker++;
		break;
	}
	case containerEventComplete:
	{
		std::cout << '\n' << operation << ": All containers populated.\n";
		this->mContainerEventTracker = 0;
		eventComplete = true;
		break;
	}
	default: {assert(&& "We've no matching event!(ContainerEvents)."); break; }
	}

	return eventComplete;
}
const bool OperationEventHandler::arithmeticEvent(const std::string& operation)
{
	using enum ArithmeticEvents;
	bool eventComplete{ false };

	switch (static_cast<ArithmeticEvents>(this->mArithmeticEventTracker))
	{
	case operationStart:
	{
		std::cout << '\n' << operation << ": Starting operation.\n";
		mArithmeticEventTracker++;
		break;
	}
	case arithmeticEventComplete:
	{
		std::cout << '\n' << operation << ": Operation complete.\n";
		this->mArithmeticEventTracker = 0;
		eventComplete = true;
		break;
	}
	default: {assert(&& "We've no matching event!(ArithmeticEvents)."); break; }
	}

	return eventComplete;
}
const bool OperationEventHandler::validationEvent(const std::string& operation) 
{
	using enum ValidationEvents;
	bool eventComplete{ false };

	switch (static_cast<ValidationEvents>(this->mValidationEventTracker))
	{
	case validationStart:
	{
		std::cout << '\n' << operation << ": Starting result validation.\n";
		mValidationEventTracker++;
		break;
	}
	case validationFailedComplete:
	{
		std::cout << '\n' << operation << ": Output data does not match the expected result. Timing results will be discarded.\n";
		mValidationEventTracker++;
		break;
	}
	case validationSuccessfulComplete:
	{
		std::cout << '\n' << operation << ": Output data matches expected results. Timing results will be recorded.\n";
		mValidationEventTracker++;
		break;
	}
	case validationEventComplete:
	{
		std::cout << '\n' << operation << ": Result validation complete.\n";
		this->mValidationEventTracker = 0;
		eventComplete = true;
		break;
	}
	default: {assert(&& "We've no matching event!(ValidationEvents)."); break; }
	}

	return eventComplete;
}
const bool OperationEventHandler::outputToFileEvent(const std::string& operation) 
{
	using enum OutputToFileEvents;
	bool eventComplete{ false };

	switch (static_cast<OutputToFileEvents>(this->mOutputToFileEventTracker))
	{
	default: {assert(&& "We've no matching event!(OutputToFileEvents)."); break; }
	}

	return eventComplete;
}