#pragma once
#ifndef VECTOR_ADDITION
#define VECTOR_ADDITION

#include "ArithmeticOperation.h"
#include "ArithmeticDetails.h"

#include <vector>

using namespace ArithmeticDetails::VectorAdditionDetails;

class VectorAddition final : public ArithmeticOperation 
{
    std::vector<int> inputVecA, inputVecB, outputVec;

public:
    VectorAddition() 
        : ArithmeticOperation{ vecAddName, vecAddSamples } {}

    void launchOperation() override final;
    void setContainer(const int& sampleChoice) override final;
    void startOperationSequence(const ProgramHandler& handler) override final;
    void validateResults() override final;
};
#endif