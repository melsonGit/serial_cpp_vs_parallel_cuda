#include "../inc/VectorAddition.h"
#include "../inc/ProgramHandler.h"


void VectorAddition::startOperationSequence(const ProgramHandler& handler)
{
	setContainer(handler.getInput());
	launchOperation();
	validateResults();
}

void VectorAddition::setContainer(const int& sampleChoice)
{
	populateContainer(this->inputVecA, this->inputVecB); // change to allow vecAdd types
}
void VectorAddition::launchOperation()
{
}
void VectorAddition::validateResults() 
{
}