#pragma once
#ifndef OPERATION_RESULT_HANDLER
#define OPERATION_RESULT_HANDLER

#include "OperationTimer.h"

class OperationResultHandler
{
private:

	const std::string mResultFilePath{ "results/" };
	const std::string mFileType{ ".csv" };
	const std::string mResultFileName{};

	const class ArithmeticOperation* ArithmeticOperationPtr;
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
		: ArithmeticOperationPtr{ &arithOp }, OperationEventHandlerPtr{ &opEventHandler }, 
		OperationTimerPtr{ &opTimer }, mResultFileName{ mResultFilePath + opName + mFileType } {}

	const std::string getResultFilePath() const;
	const std::string getResultFileName() const;

	void processOperationResults();
};
#endif