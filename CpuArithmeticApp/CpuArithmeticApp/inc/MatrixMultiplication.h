#pragma once
#ifndef MATRIX_MULTIPLICATION
#define MATRIX_MULTIPLICATION

#include "ArithmeticOperation.h"
#include "ArithmeticDetails.h"

using namespace ArithmeticDetails::MatrixMultiplicationDetails;

class MatrixMultiplication final : public ArithmeticOperation
{
public:
    MatrixMultiplication()
        : ArithmeticOperation{ matMultiName, matMultiSamples } {}

    void launchOperation() override final;
    void setContainer(const int& sampleChoice) override final;
    void startOperationSequence() override final;
    void validateResults() override final;
};
#endif