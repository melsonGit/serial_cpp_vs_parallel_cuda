#include "../inc/VectorAddition.h"

#include <vector>

void VectorAddition::startOperationSequence()
{
	setContainer(1);
	launchOperation();
	validateResults();
}

void VectorAddition::setContainer(const int& sampleChoice)
{
	switch (sampleChoice)
	{
	case 1: { this->mVAInputVecA.resize(mSampleSizes[0]); this->mVAInputVecB.resize(mSampleSizes[0]); this->mVAOutputVec.resize(mSampleSizes[0]); break; }
	case 2: { this->mVAInputVecA.resize(mSampleSizes[1]); this->mVAInputVecB.resize(mSampleSizes[1]); this->mVAOutputVec.resize(mSampleSizes[1]); break; }
	case 3: { this->mVAInputVecA.resize(mSampleSizes[2]); this->mVAInputVecB.resize(mSampleSizes[2]); this->mVAOutputVec.resize(mSampleSizes[2]); break; }
	case 4: { this->mVAInputVecA.resize(mSampleSizes[3]); this->mVAInputVecB.resize(mSampleSizes[3]); this->mVAOutputVec.resize(mSampleSizes[3]); break; }
	case 5: { this->mVAInputVecA.resize(mSampleSizes[4]); this->mVAInputVecB.resize(mSampleSizes[4]); this->mVAOutputVec.resize(mSampleSizes[4]); break; }
	default:
		// Something needs to go here....
		break;
	}

	populateContainer(this->mVAInputVecA, this->mVAInputVecB);
}

void VectorAddition::launchOperation()
{
}

void VectorAddition::validateResults() 
{
}