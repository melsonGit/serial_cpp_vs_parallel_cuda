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

    std::vector<std::size_t> mMMInputVecA, mMMInputVecB, mMMOutputVec;

    void setContainer(const int& userInput) override final;
    void launchOp() override final;
    void validateResults() override final;

public:

    MatrixMultiplication()
        : ArithmeticOperation{ matMultiName, matMultiSamples } {}
};
#endif