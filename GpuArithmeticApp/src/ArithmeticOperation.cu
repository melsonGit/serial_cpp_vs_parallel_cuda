#include "../inc/ArithmeticOperation.cuh"

void ArithmeticOperation::storeResults()
{
	this->OperationResultHandler.processOperationResults();
}

void ArithmeticOperation::updateEventHandler(const EventDirectives& event)
{
	this->OperationEventHandler.setEvent(event);
	this->OperationEventHandler.processEvent();
}

// Container Check

bool ArithmeticOperation::isNewContainer()
{
	constexpr int firstRun{ 99 }; return (this->getVecIndex() == firstRun); // First run check - any number outside 0 - 6 is fine but just to be safe
}
bool ArithmeticOperation::isContainerSameSize(const int& newIndex)
{
	return (this->getVecIndex() == newIndex); // If the same, no changes are needed
}
bool ArithmeticOperation::isContainerSmallerSize(const int& newIndex)
{
	return (this->getVecIndex() < newIndex); // If current sample selection is higher than previous run - resize()
}
bool ArithmeticOperation::isContainerLargerSize(const int& newIndex)
{
	return (this->getVecIndex() > newIndex); // If current sample selection is lower than previous run - resize() and then shrink_to_fit().
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
	//this->storeResults();
}

// CUDA Specific Functions
void ArithmeticOperation::update1DMemSize()
{
	this->mMemSize = sizeof(size_t) * this->mCurrSampleSize;
}
void ArithmeticOperation::update2DMemSize()
{
	this->mMemSize = this->tempConSize * this->tempConSize * sizeof(std::size_t);
}
void ArithmeticOperation::update1DBlockSize()
{
	this->mBLOCKS = (this->mCurrSampleSize + this->mTHREADS - 1) / this->mTHREADS;
}
void ArithmeticOperation::update2DBlockSize()
{
	this->mBLOCKS = (this->tempConSize + this->mTHREADS - 1) / this->mTHREADS;
}
void ArithmeticOperation::prep1DKernelVars()
{
	this->update1DBlockSize();
	this->update1DMemSize();
}
void ArithmeticOperation::prep2DKernelVars()
{
	this->update2DBlockSize();
	this->update2DMemSize();
}
void ArithmeticOperation::updateDimStructs()
{
	// Use dim3 structs for BLOCKS and THREADS dimensions | Passed to kernal lauch as launch arguments
	dim3 mTempThreads(this->mTHREADS, this->mTHREADS);
	dim3 mTempBlocks(this->mBLOCKS, this->mBLOCKS);

	this->mDimBlocks = mTempBlocks;
	this->mDimThreads = mTempThreads;
}