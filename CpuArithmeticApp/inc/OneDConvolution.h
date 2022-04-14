#pragma once
#ifndef ONE_CONVOLUTION
#define ONE_CONVOLUTION

#include "ArithmeticOperation.h"
#include "ArithmeticDetails.h"

#include <vector>

using namespace ArithmeticDetails::OneDConvolutionDetails;

class OneDConvolution final : public ArithmeticOperation
{
private:

    std::vector<std::size_t> mOCInputVec, mOCMaskVec, mOCOutputVec;

    void setContainer(const int& userInput) override final;
    void launchOp() override final;
    void validateResults() override final;

    template<typename P1> void populateContainer(std::vector<P1>& vecToPop);
    template<typename P1, typename ... Args> void populateContainer(std::vector<P1>& vecToPop, Args&... args);

public:

    OneDConvolution()
        : ArithmeticOperation{ oneDConvName, oneDConvSamples } {}
};

template<typename P1>
void OneDConvolution::populateContainer(std::vector<P1>& vecToPop)
{
    ArithmeticOperation::populateContainer<P1>(vecToPop);
}

template<typename P1, typename ... Args>
void OneDConvolution::populateContainer(std::vector<P1>& vecToPop, Args&... args)
{
    populateContainer(vecToPop);
    populateContainer(args...);
}
#endif