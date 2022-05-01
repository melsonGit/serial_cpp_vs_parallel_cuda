#pragma once
#ifndef OPERATION_RESULT_HANDLER
#define OPERATION_RESULT_HANDLER

#include "OperationTimer.h"

class OperationResultHandler
{
private:

	const std::string resultFilePath{ "results/" };
	const std::string fileType{ ".csv" };
	const std::string resultFileName{};

	const class ArithmeticOperation* ArithemticOperationPtr;
	class OperationEventHandler* OperationEventHandlerPtr;
	const OperationTimer* OperationTimerPtr;

	const bool doesResultDirectoryExist() const;
	const bool doesResultFileExist() const;
	void createResultDirectory() const;
	void createResultFile();
	void recordResults();
	void checkSlowTime();
	void insertSlowTime();
	void changeAvgTime();
	void checkFastTime();
	void insertFastTime();
	void insertLastestTime();

public:

	OperationResultHandler(const ArithmeticOperation& arithOp, OperationEventHandler& opEventHandler, const OperationTimer& opTimer, const std::string opName)
		: ArithemticOperationPtr{ &arithOp }, OperationEventHandlerPtr{ &opEventHandler }, 
		OperationTimerPtr{ &opTimer }, resultFileName{ resultFilePath + opName + fileType } {}

	const std::string getResultFilePath() const;
	const std::string getResultFileName() const;

	void processOperationResults();
};
#endif