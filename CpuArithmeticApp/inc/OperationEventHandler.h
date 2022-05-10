#pragma once
#ifndef OPERATION_EVENT_HANDLER
#define OPERATION_EVENT_HANDLER

#include "OperationTimer.h"
#include "EventDirectives.h"

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

	const std::unordered_map<EventDirectives, std::string> mEventHolder{};
	EventDirectives mEventId{};

	const bool checkEventExists() const;
	const bool isResultsValidatedEvent() const;
	const std::string& getEventString() const;
	void outputTimeResults() const;
	void badEvent() const;
	void outputEvent() const;
	void processTimeResults();

public:

	OperationEventHandler(const ArithmeticOperation& arithOp, const OperationResultHandler& opResultHandler, const OperationTimer& opTimer)
		: ArithmeticOperationPtr{ &arithOp }, OperationResultHandlerPtr{ &opResultHandler }, OperationTimerPtr{ &opTimer }, mEventHolder{ eventDirectiveMap } {}

	void setEvent(const EventDirectives& event);
	void processEvent() const;
};
#endif