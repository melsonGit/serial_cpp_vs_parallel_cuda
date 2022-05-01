#include "../inc/ArithmeticOperation.h"

#include <chrono>
#include <fstream>
#include <iostream>
#include <filesystem>

void ArithmeticOperation::recordResults() // this should be its own class
{
	const std::string resultFilePath{ "results/" };
	const std::string resultFile{ resultFilePath + this->getOpName() + ".csv" };

	// Check to see if results directory exists
	if (!std::filesystem::exists(resultFilePath))
	{
		std::cout << resultFilePath << " not found! Creating new " << resultFilePath << " directory\n\n"; // move to EventHandler
		std::filesystem::create_directory(resultFilePath);
	}
	else
		std::cout << resultFilePath << " found!\n\n";

	// Check to see if our operation result file exists
	if (!std::filesystem::exists(resultFile))
	{
		std::cout << "Result file doesn't exist! Creating new results file\n\n";

		std::ofstream newFile(resultFile);

		int sampleSizes{ 0 };

		newFile << "Sample Size, Slowest Time,,,,Average Time,,,,Fastest Time,,,,Lastest Time\n,us,ms,s,,us,ms,s,,us,ms,s,,us,ms,s\n"
			<< this->getOpSampleSize(sampleSizes++) << '\n' << this->getOpSampleSize(sampleSizes++) << '\n'
			<< this->getOpSampleSize(sampleSizes++) << '\n' << this->getOpSampleSize(sampleSizes++) << '\n'
			<< this->getOpSampleSize(sampleSizes++);

		newFile.close();
	}

	std::cout << "Storing " << this->getOpName() << " results into " << resultFile << ".\n\n";

	std::ofstream existingFile;
	existingFile.open(resultFile, std::fstream::app);

	existingFile << "\nHello, I aready exist! Time: " << std::chrono::system_clock::now() << '\n';
	existingFile.close();

	//this->OperationTimer.getElapsedMicroseconds();
	//this->OperationTimer.getElapsedMilliseconds();
	//this->OperationTimer.getElapsedSeconds();

	//void insertSlowTime(const std::size_t sampleSize, const int time);
	//void insertAvgTime(const std::size_t sampleSize, const int time);
	//void insertFastTime(const std::size_t sampleSize, const int time);
	//void insertLastestTime(const std::size_t sampleSize, const int time);
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