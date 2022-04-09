#pragma once
#ifndef MATRIX_MULTIPLICATION
#define MATRIX_MULTIPLICATION

#include "ArithmeticOperation.h"
#include "ArithmeticDetails.h"

#include <vector>

using namespace ArithmeticDetails::MatrixMultiplicationDetails;

class MatrixMultiplication final : public ArithmeticOperation
{
private:

    std::vector<int> mMMInputVecA, mMMInputVecB, mMMOutputVec;

    void setContainer(const int& sampleChoice) override final;
    void launchOperation() override final;
    void validateResults() override final;

public:

    MatrixMultiplication()
        : ArithmeticOperation{ matMultiName, matMultiSamples } {}

    void startOperationSequence() override final;
};
#endif