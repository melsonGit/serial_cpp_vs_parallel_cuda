#pragma once
#ifndef VECTOR_ADDITION
#define VECTOR_ADDITION

#include "ArithmeticOperation.h"
#include "ArithmeticDetails.h"

using namespace ArithmeticDetails::VectorAdditionDetails;

class VectorAddition final : public ArithmeticOperation 
{
public:
    VectorAddition() 
        : ArithmeticOperation{ vecAddName, vecAddSamples } {}

    void launchOperation() override final;
    void setContainer(const int& sampleChoice) override final;
    void startOperationSequence() override final;
    void validateResults() override final;
};
#endif