#pragma once
#ifndef VECTOR_ADDITION
#define VECTOR_ADDITION

#include "ArithmeticOperation.h"

class VectorAddition final : public ArithmeticOperation 
{
public:
    VectorAddition(const std::string& name = "Vector Addition") 
        : ArithmeticOperation{ name } {}

    void populateSampleArr() override final;
    void launchOperation() override final;
    void setContainer() override final;
    void startOperationSequence() override final;
    void validateResults() override final;
};
#endif