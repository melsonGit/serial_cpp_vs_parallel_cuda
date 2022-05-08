#pragma once
#ifndef ONE_CONVOLUTION
#define ONE_CONVOLUTION

#include "ArithmeticOperation.h"
#include "ArithmeticDetails.h"
#include "MaskAttributes.h"

#include <vector>

using namespace ArithmeticDetails::OneDConvolutionDetails;
using namespace MaskAttributes;

class OneDConvolution final : public ArithmeticOperation
{
private:

    std::vector<std::size_t> mOCInputVec, mOCMaskVec, mOCOutputVec;

    void setContainer(const int& userInput) override final;
    void launchOp() override final;
    void validateResults() override final;

    // populateContainer - 1D
    template<typename P1> void populateContainer(std::vector<P1>& vecToPop);
    template<typename P1, typename ... Args> void populateContainer(std::vector<P1>& vecToPop, Args&... args);
    // resizeContainer - 1D
    template<typename P1> void resizeContainer(const P1& newSize, std::vector<P1>& vecToResize);
    template<typename P1, typename ... Args> void resizeContainer(const P1& newSize, std::vector<P1>& vecToResize, Args&... args);
    // shrinkContainer - 1D
    template<typename P1> void shrinkContainer(std::vector<P1>& vecToShrink);
    template<typename P1, typename ... Args> void shrinkContainer(std::vector<P1>& vecToShrink, Args&... args);

public:

    OneDConvolution()
        : ArithmeticOperation{ oneDConvName, oneDConvSamples, oneDConvMaskStatus } {}
};

// populateContainer - 1D
template<typename P1>
void OneDConvolution::populateContainer(std::vector<P1>& vecToPop)
{
    ArithmeticOperation::populateContainer<P1>(vecToPop);
}
template<typename P1, typename ... Args>
void OneDConvolution::populateContainer(std::vector<P1>& vecToPop, Args&... args)
{
    this->populateContainer(vecToPop);
    this->populateContainer(args...);
}
// resizeContainer - 1D
template<typename P1>
void OneDConvolution::resizeContainer(const P1& newSize, std::vector<P1>& vecToResize)
{
    ArithmeticOperation::resizeContainer<P1>(newSize, vecToResize);
}
template<typename P1, typename ... Args>
void OneDConvolution::resizeContainer(const P1& newSize, std::vector<P1>& vecToResize, Args&... args)
{
    this->resizeContainer(newSize, vecToResize);
    this->resizeContainer(newSize, args...);
}
// shrinkContainer - 1D
template<typename P1>
void OneDConvolution::shrinkContainer(std::vector<P1>& vecToShrink)
{
    ArithmeticOperation::shrinkContainer(vecToShrink);
}
template<typename P1, typename ... Args>
void OneDConvolution::shrinkContainer(std::vector<P1>& vecToShrink, Args&... args)
{
    this->shrinkContainer(vecToShrink);
    this->shrinkContainer(args...);
}
#endif