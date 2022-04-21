#include "../inc/OperationEventHandler.h"

#include <iostream>
#include <string>
#include <cassert>

using namespace OperationEvents;

/*
	=====****| Parameter Legend |****=====

	operation:        Denotes the operation we're processing throughout events
	hasMask:		  Denotes if the operation uses a mask vector in ther arithmetic operation
	passedValidation: Denotes if the operation result validation was successful
*/

// This function is called by ArithmeticOperation object functions throughout various stages of operation execution

void OperationEventHandler::processEvent(const std::string& operation, const bool& hasMask, const bool& passedValidation)
{
	using enum EventTriggers;

	switch (static_cast<EventTriggers>(this->mEventController))
	{
	case startSetContainerEvent: {eventSetContainer(operation, hasMask); break; }
	case startLaunchOpEvent: {eventLaunchOp(operation); break; }
	case startValidationEvent: {eventValidateResults(operation, passedValidation); break; }
	//case startOutputToFileEvent: {eventOutputToFile(operation); break; }
	default: {const bool isBadTrigger{ true }; assert(!isBadTrigger && "We've no event to process!(OperationEventHandler->processEvent())."); break; }
	}
}

/*
	We enter these functions through processEvent()
	Corresponding events have a controller, which determines what process we print out
	Controller movement is determined by specific arithmetic operation execution flow
	e.g. if the operation uses a mask, if it fails result validation etc.
*/

void OperationEventHandler::eventSetContainer(const std::string& operation, const bool& hasMask)
{
	using enum ContainerEvents;

	switch (static_cast<ContainerEvents>(this->mContainerEventController))
	{
	case genericPopulationStart:
	{
		std::cout << '\n' << operation << ": Populating main containers.\n";
		this->mContainerEventController++;
		break;
	}
	case genericPopulationComplete:
	{
		std::cout << '\n' << operation << ": Main container population complete.\n";

		// We want to skip processing mask population event if our operation doesn't have a mask vector
		int skipEvent{ 3 };

		if (hasMask)
			this->mContainerEventController++;
		else
			this->mContainerEventController += skipEvent;
		break;
	}
	case maskPopulationStart:
	{
		std::cout << '\n' << operation << ": Populating mask containers.\n";
		this->mContainerEventController++;
		break;
	}
	case maskPopulationComplete:
	{
		std::cout << '\n' << operation << ": Mask container population complete.\n";
		this->mContainerEventController++;
		break;
	}
	case containerEventComplete:
	{
		std::cout << '\n' << operation << ": All containers populated.\n";
		this->resetContainerEventController();
		this->mEventController = static_cast<int>(EventTriggers::startLaunchOpEvent);
		break;
	}
	default: {const bool isBadTrigger{ true }; assert(!isBadTrigger && "We've no matching event!(eventSetContainer)."); break; }
	}
}
void OperationEventHandler::eventLaunchOp(const std::string& operation)
{
	using enum LaunchOpEvents;

	switch (static_cast<LaunchOpEvents>(this->mLaunchOpEventController))
	{
	case operationStart:
	{
		std::cout << '\n' << operation << ": Starting operation.\n";
		this->mLaunchOpEventController++;
		break;
	}
	case arithmeticEventComplete:
	{
		std::cout << '\n' << operation << ": Operation complete.\n";
		this->resetLaunchOpEventController();
		this->mEventController = static_cast<int>(EventTriggers::startValidationEvent);
		break;
	}
	default: {const bool isBadTrigger{ true }; assert(!isBadTrigger && "We've no matching event!(eventLaunchOp)."); break; }
	}
}
void OperationEventHandler::eventValidateResults(const std::string& operation, const bool& passedValidation)
{
	using enum ValidationEvents;

	switch (static_cast<ValidationEvents>(this->mValidationEventController))
	{
	case validationStart:
	{
		std::cout << '\n' << operation << ": Starting result validation.\n";
		this->mValidationEventController++;
		break;
	}
	case validationEventComplete:
	{
		std::cout << '\n' << operation << ": Result validation complete.\n";

		if (passedValidation)
			std::cout << '\n' << operation << ": Output data matches expected results.\n\t\t Timing results will be recorded.\n";
		else
			std::cout << '\n' << operation << ": Output data does not match the expected result. Timing results will be discarded.\n";

		this->resetValidationEventController();
		//this->mEventController = static_cast<int>(EventTriggers::startOutputToFileEvent); uncomment when we implement the feature
		this->resetEventController(); // remove when the above is implemented
		break;
	}
	default: {const bool isBadTrigger{ true }; assert(!isBadTrigger && "We've no matching event!(eventValidateResults)."); break; }
	}
}
#if 0
void OperationEventHandler::eventOutputToFile(const std::string& operation)
{
	using enum OutputToFileEvents;

	resetEventController();

	switch (static_cast<OutputToFileEvents>(this->mOutputToFileEventController))
	{
	default: {assert(&& "We've no matching event!(eventOutputToFile)."); break; }
	}
}
#endif

// Controller resetters

void OperationEventHandler::resetEventController()
{
	this->mEventController = 0;
}
void OperationEventHandler::resetContainerEventController()
{
	this->mContainerEventController = 0;
}
void OperationEventHandler::resetLaunchOpEventController()
{
	this->mLaunchOpEventController = 0;
}
void OperationEventHandler::resetValidationEventController()
{
	this->mValidationEventController = 0;
}