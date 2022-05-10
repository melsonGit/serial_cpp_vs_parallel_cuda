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
    void processContainerSize(const int& newIndex) override final;

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

    VectorAddition() 
        : ArithmeticOperation{ vecAddName, vecAddSamples, vecAddMaskStatus } {}
};

// populateContainer - 1D
template<typename P1>
void VectorAddition::populateContainer(std::vector<P1>& vecToPop)
{
    ArithmeticOperation::populateContainer<P1>(vecToPop);
}
template<typename P1, typename ... Args>
void VectorAddition::populateContainer(std::vector<P1>& vecToPop, Args&... args)
{
    this->populateContainer(vecToPop);
    this->populateContainer(args...);
}
// resizeContainer - 1D
template<typename P1> 
void VectorAddition::resizeContainer(const P1& newSize, std::vector<P1>& vecToResize)
{
    ArithmeticOperation::resizeContainer<P1>(newSize, vecToResize);
}
template<typename P1, typename ... Args> 
void VectorAddition::resizeContainer(const P1& newSize, std::vector<P1>& vecToResize, Args&... args)
{
    this->resizeContainer(newSize, vecToResize);
    this->resizeContainer(newSize, args...);
}
// shrinkContainer - 1D
template<typename P1> 
void VectorAddition::shrinkContainer(std::vector<P1>& vecToShrink)
{
    ArithmeticOperation::shrinkContainer(vecToShrink);
}
template<typename P1, typename ... Args> 
void VectorAddition::shrinkContainer(std::vector<P1>& vecToShrink, Args&... args)
{
    this->shrinkContainer(vecToShrink);
    this->shrinkContainer(args...);
}
#endif