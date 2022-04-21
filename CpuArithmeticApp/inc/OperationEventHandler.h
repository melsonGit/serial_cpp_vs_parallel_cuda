#pragma once
#ifndef OPERATION_EVENT_HANDLER
#define OPERATION_EVENT_HANDLER

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
	enum class LaunchOpEvents
	{
		operationStart,
		arithmeticEventComplete,
	};
	enum class ValidationEvents
	{
		validationStart,
		validationEventComplete,
	};
	//enum class OutputToFileEvents {};
	enum class EventTriggers
	{
		startSetContainerEvent,
		startLaunchOpEvent,
		startValidationEvent,
		//startOutputToFileEvent,
	};
}

class OperationEventHandler
{
private:

	// Navigates and determines events to be processed in processEvent()
	int mEventController{ 0 };
	
	// Navigates specific events pertaining to corresponding name 
	// e.g. mContainerEventController controls eventSetContainer() flow
	int mContainerEventController{ 0 };
	int mLaunchOpEventController{ 0 };
	int mValidationEventController{ 0 };
	//int mOutputToFileEventController{ 0 };

	void eventSetContainer(const std::string& operation, const bool& hasMask);
	void eventLaunchOp(const std::string& operation);
	void eventValidateResults(const std::string& operation, const bool& passedValidation);
	//void eventOutputToFile(const std::string& operation);

	// Controller resetters
	void resetEventController();
	void resetContainerEventController();
	void resetLaunchOpEventController();
	void resetValidationEventController();

public:

	OperationEventHandler() = default;

	void processEvent(const std::string& operation, const bool& hasMask = false, const bool& passedValidation = true);
};
#endif