#include "../inc/OperationEventHandler.h"
#include "../inc/ArithmeticOperation.cuh"
#include "../inc/OperationTimeHandler.h"

const bool OperationEventHandler::checkEventExists() const
{
	return (this->mEventHolder.contains(this->mEventId));
}

const bool OperationEventHandler::isResultsValidatedEvent() const
{
	return (this->mEventId == EventDirectives::resultsValidated);
}

void OperationEventHandler::badEvent() const
{
	const bool isBadTrigger{ true }; assert(!isBadTrigger && "We've no matching event!");
}

void OperationEventHandler::outputEvent() const
{
	std::cout << '\n' << this->ArithmeticOperationPtr->getOpName() << ": " << getEventString() << '\n';
}

void OperationEventHandler::processEvent() const
{
	if (this->checkEventExists())
		this->outputEvent();
	else this->badEvent();

	if (isResultsValidatedEvent())
		outputTimeResults();
}

void OperationEventHandler::outputTimeResults() const
{
	std::cout << "\n\nCPU " << this->ArithmeticOperationPtr->getOpName() << " computation time (container size : "
		<< this->ArithmeticOperationPtr->getCurrSampleSize() << ") :\n"
		<< this->OperationTimerPtr->getElapsedMicroseconds() << " us (microseconds)\n"
		<< this->OperationTimerPtr->getElapsedMilliseconds() << " ms (milliseconds)\n"
		<< this->OperationTimerPtr->getElapsedSeconds() << " s (seconds)\n\n";

	if (this->ArithmeticOperationPtr->getValidationStatus())
		std::cout << "Output passed validation. Timing results will now be recorded.\nPress any key to record results and return to sample selection menu.\n\n";
	else
		std::cout << "Output failed validation. Timing results discarded.\nPress any key to return to sample selection menu.\n\n";

	// move into function
	std::cin.get();
	std::cin.ignore();
}

const std::string& OperationEventHandler::getEventString() const
{
	auto itr = this->mEventHolder.find(this->mEventId); return itr->second;
}

void OperationEventHandler::setEvent(const EventDirectives& event)
{
	this->mEventId = event;
}