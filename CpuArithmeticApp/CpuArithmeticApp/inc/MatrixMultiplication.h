#pragma once
#ifndef MATRIX_MULTIPLICATION
#define MATRIX_MULTIPLICATION

#include "ArithmeticOperation.h"
#include "ArithmeticSampleSizes.h"

using namespace ArithmeticSampleSizes::MatrixMultiplicationSamples;

class MatrixMultiplication final : public ArithmeticOperation
{
public:
    MatrixMultiplication(const std::string& name = "Matrix Multiplication")
        : ArithmeticOperation{ name, matMultiSamples } {}

    void launchOperation() override final;
    void setContainer(const int& sampleChoice) override final;
    void startOperationSequence() override final;
    void validateResults() override final;
};
#endif