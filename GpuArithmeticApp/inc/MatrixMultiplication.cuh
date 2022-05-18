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

    std::vector<std::size_t> mMMHostInputVecA, mMMHostInputVecB, mMMHostOutputVec;

    void setContainer(const int& userInput) override final;
    void launchOp() override final;
    void validateResults() override final;
    void processContainerSize(const int& newIndex) override final;

    // Temp measures to fix GPU issues
    void tempConSizeInitTEMP(); // remove when we use 2d vectors

    // CUDA Specific Variables
    std::size_t* mMMDeviceInputVecA{ nullptr };
    std::size_t* mMMDeviceInputVecB{ nullptr };
    std::size_t* mMMDeviceOutputVec{ nullptr };

    // CUDA Specific Functions
    void allocateMemToDevice() override final;
    void copyHostToDevice() override final;
    void copyDeviceToHost() override final;
    void freeDeviceData() override final;

    // populateContainer - Native 2D
    template<typename P1> void populateContainer(std::vector<P1>& vecToPop);
    template<typename P1, typename ... Args> void populateContainer(std::vector<P1>& vecToPop, Args&... args);
    // resizeContainer - Native 2D
    template<typename P1> void resizeContainer(const P1& newSize, std::vector<P1>& vecToResize);
    template<typename P1, typename ... Args> void resizeContainer(const P1& newSize, std::vector<P1>& vecToResize, Args&... args);
    // shrinkContainer - Native 2D
    template<typename P1> void shrinkContainer(std::vector<P1>& vecToShrink);
    template<typename P1, typename ... Args> void shrinkContainer(std::vector<P1>& vecToShrink, Args&... args);

public:

    MatrixMultiplication()
        : ArithmeticOperation{ matMultiName, matMultiSamples, matMultiMaskStatus } {}
};

// populateContainer - Native 2D
template<typename P1>
void MatrixMultiplication::populateContainer(std::vector<P1>& vecToPop)
{
    ArithmeticOperation::populateContainer<P1>(vecToPop);
}
template<typename P1, typename ... Args>
void MatrixMultiplication::populateContainer(std::vector<P1>& vecToPop, Args&... args)
{
    this->populateContainer(vecToPop);
    this->populateContainer(args...);
}
// resizeContainer - Native 2D
template<typename P1>
void MatrixMultiplication::resizeContainer(const P1& newSize, std::vector<P1>& vecToResize)
{
    ArithmeticOperation::resizeContainer<P1>(newSize, vecToResize);
}
template<typename P1, typename ... Args>
void MatrixMultiplication::resizeContainer(const P1& newSize, std::vector<P1>& vecToResize, Args&... args)
{
    this->resizeContainer(newSize, vecToResize);
    this->resizeContainer(newSize, args...);
}
// shrinkContainer - Native 2D
template<typename P1>
void MatrixMultiplication::shrinkContainer(std::vector<P1>& vecToShrink)
{
    ArithmeticOperation::shrinkContainer(vecToShrink);
}
template<typename P1, typename ... Args>
void MatrixMultiplication::shrinkContainer(std::vector<P1>& vecToShrink, Args&... args)
{
    this->shrinkContainer(vecToShrink);
    this->shrinkContainer(args...);
}
#endif