#pragma once
#ifndef MATRIX_MULTIPLICATION
#define MATRIX_MULTIPLICATION

#include "ArithmeticOperation.cuh"
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
    void processContainerSize(const int& newIndex) override final;

    // CUDA Specific Functions
    void allocateMemToDevice() override final;
    void copyHostToDevice() override final;
    void copyDeviceToHost() override final;
    void freeDeviceData() override final;

    // populateContainer - 2D
    template<typename P1> void populateContainer(std::vector<std::vector<P1>>& vecToPop);
    template<typename P1, typename ... Args> void populateContainer(std::vector<std::vector<P1>>& vecToPop, Args&... args);
    // resizeContainer - 2D
    template<typename P1> void resizeContainer(const P1& newSize, std::vector<std::vector<P1>>& vecToResize);
    template<typename P1, typename ... Args> void resizeContainer(const P1& newSize, std::vector<std::vector<P1>>& vecToResize, Args&... args);
    // shrinkContainer - 2D
    template<typename P1> void shrinkContainer(std::vector<std::vector<P1>>& vecToShrink);
    template<typename P1, typename ... Args> void shrinkContainer(std::vector<std::vector<P1>>& vecToShrink, Args&... args);

public:

    MatrixMultiplication()
        : ArithmeticOperation{ matMultiName, matMultiSamples, matMultiMaskStatus } {}
};

// populateContainer - 2D
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
// resizeContainer - 2D
template<typename P1>
void MatrixMultiplication::resizeContainer(const P1& newSize, std::vector<std::vector<P1>>& vecToResize)
{
    ArithmeticOperation::resizeContainer<P1>(newSize, vecToResize);
}
template<typename P1, typename ... Args>
void MatrixMultiplication::resizeContainer(const P1& newSize, std::vector<std::vector<P1>>& vecToResize, Args&... args)
{
    this->resizeContainer(newSize, vecToResize);
    this->resizeContainer(newSize, args...);
}
// shrinkContainer - 2D
template<typename P1> 
void MatrixMultiplication::shrinkContainer(std::vector<std::vector<P1>>& vecToShrink)
{
    ArithmeticOperation::shrinkContainer(vecToShrink);
}
template<typename P1, typename ... Args> 
void MatrixMultiplication::shrinkContainer(std::vector<std::vector<P1>>& vecToShrink, Args&... args)
{
    this->shrinkContainer(vecToShrink);
    this->shrinkContainer(args...);
}
#endif