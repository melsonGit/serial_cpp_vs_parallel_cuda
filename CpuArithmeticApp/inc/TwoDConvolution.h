#pragma once
#ifndef TWO_CONVOLUTION
#define TWO_CONVOLUTION

#include "ArithmeticOperation.h"
#include "ArithmeticDetails.h"

#include <vector>

using namespace ArithmeticDetails::TwoDConvolutionDetails;

class TwoDConvolution final : public ArithmeticOperation
{
private:

    std::vector<std::size_t> mTCInputVec, mTCMaskVec, mTCOutputVec;

    void setContainer(const int& userInput) override final;
    void launchOp() override final;
    void validateResults() override final;
    const std::size_t tempConSizeInit(); // remove when we use 2d vectors

    // populateContainer - 1D
    template<typename P1> void populateContainer(std::vector<P1>& vecToPop);
    template<typename P1, typename ... Args> void populateContainer(std::vector<P1>&, Args&... args);
    // resizeContainer - 1D
    template<typename P1> void resizeContainer(const P1& newSize, std::vector<P1>& vecToResize);
    template<typename P1, typename ... Args> void resizeContainer(const P1& newSize, std::vector<P1>& vecToResize, Args&... args);
    // shrinkContainer - 1D
    template<typename P1> void shrinkContainer(std::vector<P1>& vecToShrink);
    template<typename P1, typename ... Args> void shrinkContainer(std::vector<P1>& vecToShrink, Args&... args);

public:

    TwoDConvolution()
        : ArithmeticOperation{ twoDConvName, twoDConvSamples, twoDConvMaskStatus } {}
};

// populateContainer - 1D
template<typename P1>
void TwoDConvolution::populateContainer(std::vector<P1>& vecToPop)
{
    ArithmeticOperation::populateContainer<P1>(vecToPop);
}
template<typename P1, typename ... Args>
void TwoDConvolution::populateContainer(std::vector<P1>& vecToPop, Args&... args)
{
    this->populateContainer(vecToPop);
    this->populateContainer(args...);
}
// resizeContainer - 1D
template<typename P1>
void TwoDConvolution::resizeContainer(const P1& newSize, std::vector<P1>& vecToResize)
{
    ArithmeticOperation::resizeContainer<P1>(newSize, vecToResize);
}
template<typename P1, typename ... Args>
void TwoDConvolution::resizeContainer(const P1& newSize, std::vector<P1>& vecToResize, Args&... args)
{
    this->resizeContainer(newSize, vecToResize);
    this->resizeContainer(newSize, args...);
}
// shrinkContainer - 1D
template<typename P1>
void TwoDConvolution::shrinkContainer(std::vector<P1>& vecToShrink)
{
    ArithmeticOperation::shrinkContainer(vecToShrink);
}
template<typename P1, typename ... Args>
void TwoDConvolution::shrinkContainer(std::vector<P1>& vecToShrink, Args&... args)
{
    this->shrinkContainer(vecToShrink);
    this->shrinkContainer(args...);
}
#endif