#include "../inc/VectorAddition.h"
#include "../inc/ProgramHandler.h"

#include <vector>

void VectorAddition::startOperationSequence(const ProgramHandler& handler)
{
	setContainer(handler.getInput());
	launchOperation();
	validateResults();
}

void VectorAddition::setContainer(const int& sampleChoice)
{
	switch (sampleChoice)
	{
	case 1:
	{
		this->mInputVecA.resize(mSampleSizes[0]);
		this->mInputVecB.resize(mSampleSizes[0]);
		this->mOutputVec.resize(mSampleSizes[0]);
		break;
	}
	case 2:
	{
		this->mInputVecA.resize(mSampleSizes[1]);
		this->mInputVecB.resize(mSampleSizes[1]);
		this->mOutputVec.resize(mSampleSizes[1]);
		break;
	}
	case 3:
	{
		this->mInputVecA.resize(mSampleSizes[2]);
		this->mInputVecB.resize(mSampleSizes[2]);
		this->mOutputVec.resize(mSampleSizes[2]);
		break;
	}
	case 4:
	{
		this->mInputVecA.resize(mSampleSizes[3]);
		this->mInputVecB.resize(mSampleSizes[3]);
		this->mOutputVec.resize(mSampleSizes[3]);
		break;
	}
	case 5:
	{
		this->mInputVecA.resize(mSampleSizes[4]);
		this->mInputVecB.resize(mSampleSizes[4]);
		this->mOutputVec.resize(mSampleSizes[4]);
		break;
	}
	default:
		// Something needs to go here....
		break;
	}

	populateContainer(this->mInputVecA, this->mInputVecB);
}

void VectorAddition::launchOperation()
{
}

void VectorAddition::validateResults() 
{
}