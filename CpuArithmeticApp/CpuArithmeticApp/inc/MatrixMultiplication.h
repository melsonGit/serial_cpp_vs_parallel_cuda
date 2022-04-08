#pragma once
#ifndef MATRIX_MULTIPLICATION
#define MATRIX_MULTIPLICATION

#include "ArithmeticOperation.h"
#include "ArithmeticDetails.h"
#include "ProgramHandler.h"

using namespace ArithmeticDetails::MatrixMultiplicationDetails;

class MatrixMultiplication final : public ArithmeticOperation
{
private:

    void launchOperation() override final;
    void setContainer(const int& sampleChoice) override final;
    void validateResults() override final;

public:

    MatrixMultiplication()
        : ArithmeticOperation{ matMultiName, matMultiSamples } {}

    void startOperationSequence(const ProgramHandler& handler) override final;
};
#endif