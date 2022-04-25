#pragma once
#ifndef OPERATION_EVENT_HANDLER
#define OPERATION_EVENT_HANDLER

#include "OperationTimer.h"

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
		startOutputToFileEvent,
	};
}

class OperationEventHandler
{
private:

	const class ArithmeticOperation* ArithemticOperationPtr;
	const OperationTimer* OperationTimerPtr;

	// Navigates and determines events to be processed in processEvent()
	int mMainEventController{ 0 };
	
	// Navigates specific events pertaining to corresponding name 
	// e.g. mContainerEventController controls eventSetContainer() flow
	int mContainerEventController{ 0 };
	int mLaunchOpEventController{ 0 };
	int mValidationEventController{ 0 };
	int mOutputToFileEventController{ 0 };

	void eventSetContainer();
	void eventLaunchOp();
	void eventValidateResults();
	//void eventOutputToFile();

	// Controller resetters
	void resetMainEventController();
	void resetContainerEventController();
	void resetLaunchOpEventController();
	void resetValidationEventController();

public:

	OperationEventHandler(const ArithmeticOperation& arithOp, const OperationTimer& opTimer)
		: ArithemticOperationPtr{ &arithOp }, OperationTimerPtr{ &opTimer } {}

	void processEvent();
};
#endif