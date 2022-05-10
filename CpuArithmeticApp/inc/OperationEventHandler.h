#pragma once
#ifndef OPERATION_EVENT_HANDLER
#define OPERATION_EVENT_HANDLER

#include "OperationTimer.h"
#include "EventArchive.h"

#include <cassert>
#include <iostream>
#include <vector>
#include <unordered_map>

class OperationEventHandler
{
private:

	const class ArithmeticOperation* ArithmeticOperationPtr;
	const class OperationResultHandler* OperationResultHandlerPtr;
	const OperationTimer* OperationTimerPtr;

	const std::unordered_map<OperationEvents, std::string> eventHolder{};
	OperationEvents eventId{};

	const bool checkEventExists() const;
	const bool isResultsValidatedEvent() const;
	const std::string& getEventString() const;
	void outputTimeResults() const;
	void badEvent() const;
	void outputEvent() const;
	void processTimeResults();

public:

	OperationEventHandler(const ArithmeticOperation& arithOp, const OperationResultHandler& opResultHandler, const OperationTimer& opTimer)
		: ArithmeticOperationPtr{ &arithOp }, OperationResultHandlerPtr{ &opResultHandler }, OperationTimerPtr{ &opTimer }, eventHolder{ eventArchive } {}

	void setEvent(const OperationEvents& event);
	void processEvent() const;
};
#endif