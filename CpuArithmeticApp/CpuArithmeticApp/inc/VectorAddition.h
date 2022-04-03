#pragma once
#ifndef VECTOR_ADDITION
#define VECTOR_ADDITION

#include "ArithmeticOperation.h"
#include "ArithmeticSampleSizes.h"

using namespace ArithmeticSampleSizes::VectorAdditionSamples;

class VectorAddition final : public ArithmeticOperation 
{
public:
    VectorAddition(const std::string& name = "Vector Addition") 
        : ArithmeticOperation{ name, vecAddSamples } {}

    void launchOperation() override final;
    void setContainer(const int& sampleChoice) override final;
    void startOperationSequence() override final;
    void validateResults() override final;
};
#endif