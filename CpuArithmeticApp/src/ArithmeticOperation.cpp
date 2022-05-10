#include "../inc/ArithmeticOperation.h"


void ArithmeticOperation::storeResults()
{
	this->OperationResultHandler.processOperationResults();
}

void ArithmeticOperation::updateEventHandler(const EventDirectives& event)
{
	this->OperationEventHandler.setEvent(event);
	this->OperationEventHandler.processEvent();
}

// Getters

const bool& ArithmeticOperation::getValidationStatus() const
{
	return this->mHasPassedValidation;
}
const int& ArithmeticOperation::getVecIndex() const
{
	return this->mVecIndex;
}
const std::size_t& ArithmeticOperation::getCurrSampleSize() const
{
	return this->mCurrSampleSize;
}
const bool& ArithmeticOperation::getMaskStatus() const
{
	return this->mHasMask;
}
const std::string& ArithmeticOperation::getOpName() const
{
	return this->mOperationName;
}
const std::size_t& ArithmeticOperation::getOpSampleSize(const int& option) const
{
	return this->mSampleSizes[option];
}

// Setters
void ArithmeticOperation::setCurrSampleSize(const int& index)
{
	this->mCurrSampleSize = this->mSampleSizes[index];
}
void ArithmeticOperation::setValidationStatus(const bool& validationResults)
{
	this->mHasPassedValidation = validationResults;
}
void ArithmeticOperation::setVecIndex(const int& newIndex)
{
	this->mVecIndex = newIndex;
}
void ArithmeticOperation::startOpSeq(const int& userInput)
{
	this->setContainer(userInput);
	this->launchOp();
	this->validateResults();
	this->storeResults();
}