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

    std::vector<std::vector<std::size_t>> mMMInputVecA, mMMInputVecB, mMMOutputVec;

    void setContainer(const int& userInput) override final;
    void launchOp() override final;
    void validateResults() override final;

    template<typename P1> void populateContainer(std::vector<std::vector<P1>>& vecToPop);
    template<typename P1, typename ... Args> void populateContainer(std::vector<std::vector<P1>>& vecToPop, Args&... args);

public:

    MatrixMultiplication()
        : ArithmeticOperation{ matMultiName, matMultiSamples, matMultiMaskStatus } {}
};

template<typename P1>
void MatrixMultiplication::populateContainer(std::vector<std::vector<P1>>& vecToPop)
{
    ArithmeticOperation::populateContainer<P1>(vecToPop);
}

template<typename P1, typename ... Args>
void MatrixMultiplication::populateContainer(std::vector<std::vector<P1>>& vecToPop, Args&... args)
{
    this->populateContainer(vecToPop);
    this->populateContainer(args...);
}
#endif