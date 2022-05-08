#pragma once
#ifndef OPERATION_EVENT_HANDLER
#define OPERATION_EVENT_HANDLER

#include "OperationTimer.h"

namespace OperationEvents
{
	enum class ContainerEvents
	{
		containerEventStart,
		containerEventEnd,
	};
	enum class LaunchOpEvents
	{
		operationEventStart,
		arithmeticEventEnd,
	};
	enum class ValidationEvents
	{
		validationEventStart,
		validationEventEnd,
	};
	enum class OutputToFileEvents 
	{
		outputToFileStart,
		outputToFileCreateDirectoryStart,
		outputToFileCreateDirectoryComplete,
		outputToFileCreateFileStart,
		outputToFileCreateFileComplete,
		outputToFileDirFileChecksComplete,
		outputToFileRecordStart,
		outputToFileRecordComplete,
		outputToFileEventComplete,
	};
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
	const class OperationResultHandler* OperationResultHandlerPtr;
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
	void eventOutputToFile(const bool& componentPresent, const bool& isFile);

	// Controller resetters
	void resetMainEventController();
	void resetContainerEventController();
	void resetLaunchOpEventController();
	void resetValidationEventController();
	void resetOutputToFileEventController();

public:

	OperationEventHandler(const ArithmeticOperation& arithOp, const OperationResultHandler& opResultHandler, const OperationTimer& opTimer)
		: ArithemticOperationPtr{ &arithOp }, OperationResultHandlerPtr{ &opResultHandler }, OperationTimerPtr{ &opTimer } {}

	void processEvent(const bool& eventOutputToFileValidation = true, const bool& isFile = true);
};
#endif