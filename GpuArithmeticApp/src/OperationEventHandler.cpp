#include "../inc/OperationEventHandler.h"
#include "../inc/ArithmeticOperation.cuh"
#include "../inc/OperationTimeHandler.h"

bool OperationEventHandler::checkEventExists() const
{
	return (this->mEventHolder.contains(this->mEventId));
}

bool OperationEventHandler::isResultsValidatedEvent() const
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
	constexpr int widthApTime{ 10 };
	constexpr int widthApMetric{ 5 };

	std::cout << "\n\nGPU " << this->ArithmeticOperationPtr->getOpName() << " computation time (container size : "
		<< this->ArithmeticOperationPtr->getCurrSampleSize() << ") :\n\n"
		<< std::setw(widthApTime) << this->OperationTimerPtr->getElapsedSeconds() << std::setw(widthApMetric) << " s " << std::setw(widthApMetric) << "(seconds)\n"
		<< std::setw(widthApTime) << this->OperationTimerPtr->getElapsedMilliseconds() << std::setw(widthApMetric) << " ms " << std::setw(widthApMetric) << "(milliseconds)\n"
		<< std::setw(widthApTime) << this->OperationTimerPtr->getElapsedMicroseconds() << std::setw(widthApMetric) << " us " << std::setw(widthApMetric) << "(microseconds)\n\n\n";

	if (this->ArithmeticOperationPtr->getValidationStatus())
		std::cout << "Output passed validation. Please record the above timing results.\nPress any key to return to sample selection menu.\n\n";
	else
		std::cout << "Output failed validation. Please discard timing results.\nPress any key to return to sample selection menu.\n\n";

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