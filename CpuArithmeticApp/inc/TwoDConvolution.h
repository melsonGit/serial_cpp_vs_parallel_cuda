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

    template<typename P1> void populateContainer(std::vector<P1>& vecToPop);
    template<typename P1, typename ... Args> void populateContainer(std::vector<P1>&, Args&... args);

public:

    TwoDConvolution()
        : ArithmeticOperation{ twoDConvName, twoDConvSamples } {}
};

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
#endif