#pragma once
#ifndef VECTOR_ADDITION
#define VECTOR_ADDITION

#include "ArithmeticOperation.h"
#include "ArithmeticDetails.h"

#include <vector>

using namespace ArithmeticDetails::VectorAdditionDetails;

class VectorAddition final : public ArithmeticOperation 
{
private:

    std::vector<std::size_t> mVAInputVecA, mVAInputVecB, mVAOutputVec;

    void setContainer(const int& userInput) override final;
    void launchOp() override final;
    void validateResults() override final;

    template<typename P1> void populateContainer(std::vector<P1>& vecToPop);
    template<typename P1, typename ... Args> void populateContainer(std::vector<P1>& vecToPop, Args&... args);

public:

    VectorAddition() 
        : ArithmeticOperation{ vecAddName, vecAddSamples } {}
};

template<typename P1>
void VectorAddition::populateContainer(std::vector<P1>& vecToPop)
{
    ArithmeticOperation::populateContainer<P1>(vecToPop);
}

template<typename P1, typename ... Args>
void VectorAddition::populateContainer(std::vector<P1>& vecToPop, Args&... args)
{
    populateContainer(vecToPop);
    populateContainer(args...);
}
#endif