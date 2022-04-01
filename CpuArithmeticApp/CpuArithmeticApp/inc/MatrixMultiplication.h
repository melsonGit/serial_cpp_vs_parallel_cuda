#pragma once
#ifndef MATRIX_MULTIPLICATION
#define MATRIX_MULTIPLICATION

#include "ArithmeticOperation.h"

class MatrixMultiplication final : public ArithmeticOperation
{
public:
    MatrixMultiplication(const std::string& name = "Matrix Multiplication")
        : ArithmeticOperation{ name } {}

    void populateSampleArr() override final;
    void launchOperation() override final;
    void setContainer() override final;
    void startOperationSequence() override final;
    void validateResults() override final;
};
#endif