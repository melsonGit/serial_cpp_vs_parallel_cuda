#include "../inc/ArithmeticOperation.h"

#include <fstream>
#include <iostream>
#include <filesystem>

void ArithmeticOperation::recordResults() // this should be its own class
{
	const std::string resultFilePath{ "results/" };
	const std::string resultFile{ this->getOpName() + ".csv" };

	// Check to see if results directory exists
	if (!std::filesystem::exists(resultFilePath))
	{
		std::cout << "Results directory doesn't exist! Creating new results directory\n\n"; // move to EventHandler
		std::filesystem::create_directory(resultFilePath);
	}
	else
		std::cout << resultFilePath << " exists!\n\n";

	// Check to see if our operation result file exists
	if (std::filesystem::exists(resultFilePath + resultFile))
	{
		std::cout << "Results file exists!";

		std::ofstream existingFile(resultFilePath + resultFile);

		existingFile << "Hello, I aready exist!\n";
		existingFile.close();

		//this->OperationTimer.getElapsedMicroseconds();
		//this->OperationTimer.getElapsedMilliseconds();
		//this->OperationTimer.getElapsedSeconds();
	}
	else
	{

		// we should set categories here (op name, op sample size, slowest, average and fastest speed recorded)

		std::cout << "Results file doesn't Exist! Creating new results file\n\n";

		std::ofstream newFile(resultFilePath + resultFile);

		newFile << "I am " << this->getOpName() << ".\n";
		newFile.close();
	}
}
void ArithmeticOperation::bestOperationTimes()
{
	/*
		Uses name, sample size and timing results from OperationTimer
		Caches times into a hashmap (even when the program is closed), looks for results stored in files used by recordResults(),
		and checks files for each operations fastest time in each sample size range
	*/
}

// Getters

const bool ArithmeticOperation::getValidationStatus() const
{
	return this->hasPassedValidation;
}
const int ArithmeticOperation::getVecIndex() const
{
	return this->vecIndex;
}
const std::size_t ArithmeticOperation::getCurrSampleSize() const
{
	return this->currSampleSize;
}
const bool ArithmeticOperation::getMaskStatus() const
{
	return this->hasMask;
}
const std::string ArithmeticOperation::getOpName() const
{
	return this->mOperationName;
}
const std::size_t ArithmeticOperation::getOpSampleSize(const int& option) const
{
	return this->mSampleSizes[option];
}

// Setters
void ArithmeticOperation::setCurrSampleSize(const int& index)
{
	this->currSampleSize = this->mSampleSizes[index];
}
void ArithmeticOperation::setValidationStatus(const bool& validationResults)
{
	this->hasPassedValidation = validationResults;
}
void ArithmeticOperation::setVecIndex(const int& newIndex)
{
	this->vecIndex = newIndex;
}
void ArithmeticOperation::startOpSeq(const int& userInput)
{
	this->setContainer(userInput);
	this->launchOp();
	this->validateResults();
}