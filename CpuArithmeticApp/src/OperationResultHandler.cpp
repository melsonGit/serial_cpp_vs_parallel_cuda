#include "../inc/OperationResultHandler.h"
#include "../inc/ArithmeticOperation.h"
#include "../inc/OperationEventHandler.h"

#include <fstream>
#include <iostream>
#include <filesystem>

void OperationResultHandler::processOperationResults()
{
	if (!this->doesResultDirectoryExist())
	{
		const bool directoryPresent{ false };
		const bool isFile{ false };

		this->createResultDirectory();
		this->createResultFile();
	}
	else if (!this->doesResultFileExist())
	{
		const bool filePresent{ false };
		const bool isFile{ true };

		this->createResultFile();
	}
}
const bool OperationResultHandler::doesResultDirectoryExist() const
{
	return std::filesystem::exists(this->resultFilePath);
}
void OperationResultHandler::createResultDirectory() const
{
	std::filesystem::create_directory(this->resultFilePath);
}
const bool OperationResultHandler::doesResultFileExist() const
{
	return std::filesystem::exists(this->resultFileName);
}
void OperationResultHandler::createResultFile()
{
	std::ofstream newFile(this->resultFileName);

	int sampleSizes{ 0 };

	newFile << "Sample Size, Slowest Time,,,,Average Time,,,,Fastest Time,,,,Lastest Time\n,us,ms,s,,us,ms,s,,us,ms,s,,us,ms,s\n"
		<< this->ArithmeticOperationPtr->getOpSampleSize(sampleSizes++) << '\n' << this->ArithmeticOperationPtr->getOpSampleSize(sampleSizes++) << '\n'
		<< this->ArithmeticOperationPtr->getOpSampleSize(sampleSizes++) << '\n' << this->ArithmeticOperationPtr->getOpSampleSize(sampleSizes++) << '\n'
		<< this->ArithmeticOperationPtr->getOpSampleSize(sampleSizes++);

	newFile.close();
}
void OperationResultHandler::recordResults()
{
	std::cout << "Storing " << this->ArithmeticOperationPtr->getOpName() << " results into " << this->resultFileName << ".\n\n";

	std::ofstream existingFile;
	existingFile.open(resultFileName, std::fstream::app);

	existingFile.close();
}

void OperationResultHandler::checkSlowTime()
{

}
void OperationResultHandler::insertSlowTime()
{

}
void OperationResultHandler::changeAvgTime()
{

}
void OperationResultHandler::checkFastTime()
{

}
void OperationResultHandler::insertFastTime()
{

}
void OperationResultHandler::insertLastestTime()
{

}

// Getter

const std::string OperationResultHandler::getResultFilePath() const
{
	return this->resultFilePath;
}
const std::string OperationResultHandler::getResultFileName() const
{
	return this->resultFileName;
}