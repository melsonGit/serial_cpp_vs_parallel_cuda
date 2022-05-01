#include "../inc/OperationEventHandler.h"
#include "../inc/ArithmeticOperation.h"
#include "../inc/OperationTimer.h"

#include <iostream>
#include <string>
#include <cassert>

using namespace OperationEvents;

// This function is called by ArithmeticOperation object functions throughout various stages of operation execution
void OperationEventHandler::processEvent(const bool& eventOutputToFileValidation, const bool& isFile)
{
	using enum EventTriggers;

	switch (static_cast<EventTriggers>(this->mMainEventController))
	{
	case startSetContainerEvent: {eventSetContainer(); break; }
	case startLaunchOpEvent: {eventLaunchOp(); break; }
	case startValidationEvent: {eventValidateResults(); break; }
	case startOutputToFileEvent: {eventOutputToFile(eventOutputToFileValidation, isFile); break; }
	default: {const bool isBadTrigger{ true }; assert(!isBadTrigger && "We've no event to process!(OperationEventHandler->processEvent())."); break; }
	}
}

/*
	We enter these functions through processEvent()
	Corresponding events have a controller, which determines what process we control
	Controller movement is determined by specific arithmetic operation execution flow
	e.g. if the operation uses a mask, if it fails result validation etc.
*/
/*
	=====****| Calling Instructions |****=====

	processEvent() is to only be called in ArithmeticOperation children class functions setContainer(), launchOp() and validateResults();

	processEvent() assumes your operations have no mask and has passed validation (default arguments).... 
	... so please ensure processEvent() is provided overriding arguments in areas where we need to specify if:
	1 - the operation uses a mask
	2 - processEvent() is being called from within validateResults();

	eventSetContainer()
		with mask    : call processEvent() 5 times
		without mask : call processEvent() 3 times
	eventLaunchOp()
		general use  : call processEvent() 2 times
	eventValidateResults()
		general use  : call processEvent() 2 times
	eventOutputToFile()
		general use  : call processEvent() ?? times
*/

void OperationEventHandler::eventSetContainer()
{
	using enum ContainerEvents;

	switch (static_cast<ContainerEvents>(this->mContainerEventController))
	{
	case genericPopulationStart:
	{
		std::cout << '\n' << this->ArithemticOperationPtr->getOpName() << ": Populating main containers.\n";
		this->mContainerEventController++;
		break;
	}
	case genericPopulationComplete:
	{
		std::cout << '\n' << this->ArithemticOperationPtr->getOpName() << ": Main container population complete.\n";

		// We want to skip processing mask population event if our operation doesn't have a mask vector
		int skipEvent{ 3 };

		if (this->ArithemticOperationPtr->getMaskStatus())
			this->mContainerEventController++;
		else
			this->mContainerEventController += skipEvent;
		break;
	}
	case maskPopulationStart:
	{
		std::cout << '\n' << this->ArithemticOperationPtr->getOpName() << ": Populating mask containers.\n";
		this->mContainerEventController++;
		break;
	}
	case maskPopulationComplete:
	{
		std::cout << '\n' << this->ArithemticOperationPtr->getOpName() << ": Mask container population complete.\n";
		this->mContainerEventController++;
		break;
	}
	case containerEventComplete:
	{
		std::cout << '\n' << this->ArithemticOperationPtr->getOpName() << ": All containers populated.\n";
		this->resetContainerEventController();
		this->mMainEventController = static_cast<int>(EventTriggers::startLaunchOpEvent);
		break;
	}
	default: {const bool isBadTrigger{ true }; assert(!isBadTrigger && "We've no matching event!(eventSetContainer)."); break; }
	}
}
void OperationEventHandler::eventLaunchOp()
{
	using enum LaunchOpEvents;

	switch (static_cast<LaunchOpEvents>(this->mLaunchOpEventController))
	{
	case operationStart:
	{
		std::cout << '\n' << this->ArithemticOperationPtr->getOpName() << ": Starting operation.\n";
		this->mLaunchOpEventController++;
		break;
	}
	case arithmeticEventComplete:
	{
		std::cout << '\n' << this->ArithemticOperationPtr->getOpName() << ": Operation complete.\n";
		this->resetLaunchOpEventController();
		this->mMainEventController = static_cast<int>(EventTriggers::startValidationEvent);
		break;
	}
	default: {const bool isBadTrigger{ true }; assert(!isBadTrigger && "We've no matching event!(eventLaunchOp)."); break; }
	}
}
void OperationEventHandler::eventValidateResults()
{
	using enum ValidationEvents;

	switch (static_cast<ValidationEvents>(this->mValidationEventController))
	{
	case validationStart:
	{
		std::cout << '\n' << this->ArithemticOperationPtr->getOpName() << ": Starting result validation.\n";
		this->mValidationEventController++;
		break;
	}
	case validationEventComplete:
	{
		std::cout << '\n' << this->ArithemticOperationPtr->getOpName() << ": Result validation complete.\n\n";

		// Output timing to complete operation and container size
		std::cout << "CPU " << this->ArithemticOperationPtr->getOpName() << " computation time (container size : " 
			<< this->ArithemticOperationPtr->getCurrSampleSize() << ") :\n" 
			<< this->OperationTimerPtr->getElapsedMicroseconds() << " us (microseconds)\n"
			<< this->OperationTimerPtr->getElapsedMilliseconds() << " ms (milliseconds)\n"
			<< this->OperationTimerPtr->getElapsedSeconds() << " s (seconds)\n";

		if (this->ArithemticOperationPtr->getValidationStatus())
			std::cout << '\n' << this->ArithemticOperationPtr->getOpName() << ": Output data matches expected results. Timing results recorded.\n";
		else
			std::cout << '\n' << this->ArithemticOperationPtr->getOpName() << ": Output data does not match the expected result. Timing results discarded.\n\n";

		std::cout << "Press any key to continue.\n\n";
		std::cin.get();
		std::cin.ignore();

		this->resetValidationEventController();
		this->mMainEventController = static_cast<int>(EventTriggers::startOutputToFileEvent);
		break;
	}
	default: {const bool isBadTrigger{ true }; assert(!isBadTrigger && "We've no matching event!(eventValidateResults)."); break; }
	}
}
void OperationEventHandler::eventOutputToFile(const bool& componentPresent, const bool& isFile)
{
	using enum OutputToFileEvents;

	switch (static_cast<OutputToFileEvents>(this->mOutputToFileEventController))
	{
	case outputToFileStart:
	{
		if (componentPresent && isFile)
		{
			int skipEvent{ 7 };
			mOutputToFileEventController += skipEvent;

			std::cout << '\n' << this->ArithemticOperationPtr->getOpName() << ": .\n";
		}
		else if (!componentPresent && !isFile)
		{
			std::cout << '\n' << this->ArithemticOperationPtr->getOpName() << ": " << this->OperationResultHandlerPtr->getResultFilePath()
				<< " directory not found! Creating new " << this->OperationResultHandlerPtr->getResultFilePath() << " directory.\n";

			this->mOutputToFileEventController++;
		}
		else if (!componentPresent && isFile)
		{

		}

		break;
	}
	case outputToFileCreateDirectoryStart:
	{

		break;
	}
	case outputToFileCreateDirectoryComplete:
		break;
	case outputToFileCreateFileWithDirStart:
	{
		std::cout << this->ArithemticOperationPtr->getOpName() << ": " << "Result file doesn't exist! Creating and configuring new results file.\n";
		break;
	}
	case outputToFileCreateFileWithDirComplete:
		break;
	case outputToFileCreateFileStart:
		break;
	case outputToFileCreateFileComplete:
		break;
	case outputToFileDirFileChecksComplete:
		break;
	case outputToFileRecordStart:
		break;
	case outputToFileRecordComplete:
		break;
	case outputToFileEventComplete:
	{
		this->resetOutputToFileEventController();
		this->resetMainEventController();
		break;
	}
	default: {const bool isBadTrigger{ true }; assert(!isBadTrigger && "We've no matching event!(eventOutputToFile)."); break; }
	}
}

// Controller resetters

void OperationEventHandler::resetMainEventController()
{
	this->mMainEventController = 0;
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
void OperationEventHandler::resetOutputToFileEventController()
{
	this->mOutputToFileEventController = 0;
}