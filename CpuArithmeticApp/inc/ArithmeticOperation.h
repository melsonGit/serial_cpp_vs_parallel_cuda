#pragma once
#ifndef ARITHMETIC_OPERATION
#define ARITHMETIC_OPERATION

#include "OperationEventHandler.h"
#include "OperationTimeHandler.h"
#include "RandNumGen.h"
#include "MaskAttributes.h"

#include <algorithm>
#include <array>
#include <string>

class ArithmeticOperation
{
protected:

    const std::string mOperationName{};
    const std::array<std::size_t, 5> mSampleSizes{};
    const bool mHasMask{};

    // ArithmeticOperation function checks 
    bool mHasPassedValidation{}; // Used in result validation checks (refer to child validateResults())
    std::size_t mCurrSampleSize{}; // Used in container size checks (refer to child setContainer())
    int mVecIndex{ 99 }; // Default first run value (see setContainer() of any child). Any number outside 0 - 6 is fine.

    // Operation Tools
    OperationTimeHandler OperationTimeHandler{}; // Collects and provides kernel execution time
    OperationEventHandler OperationEventHandler; // Reports operation events to interface

    // Operation-specific functions
    virtual void setContainer(const int& userInput) = 0;
    virtual void launchOp() = 0;
    virtual void validateResults() = 0;

    // Container Checks
    virtual void processContainerSize(const int& newIndex) = 0; // Higher function for container functions
    bool isNewContainer();
    bool isContainerSameSize(const int& newIndex);
    bool isContainerSmallerSize(const int& newIndex);
    bool isContainerLargerSize(const int& newIndex);

    // Setters & Getters
    void setCurrSampleSize(const int& index);
    void setValidationStatus(const bool& validationResult);
    void setVecIndex(const int& newIndex);
    const int& getVecIndex() const;

    // Send events to OperationEventHandler
    void updateEventHandler(const EventDirectives& event);

    // Functions used by all operations
    // populateContainer
    template<typename P1> void populateContainer (std::vector<P1>& vecToPop);
    template<typename P1, typename ... Args> void populateContainer(std::vector<P1>& vecToPop, Args&... args);
    template<typename P1> void populateContainer (std::vector<std::vector<P1>>& vecToPop);
    template<typename P1, typename ... Args> void populateContainer(std::vector<std::vector<P1>>& vecToPop, Args&... args);
    // resizeContainer
    template<typename P1> void resizeContainer(const P1& newSize, std::vector<P1>& vecToResize);
    template<typename P1, typename ... Args> void resizeContainer(const P1& newSize, std::vector<P1>& vecToResize, Args&... args);
    template<typename P1> void resizeContainer(const P1& newSize, std::vector<std::vector<P1>>& vecToResize);
    template<typename P1, typename ... Args> void resizeContainer(const P1& newSize, std::vector<std::vector<P1>>& vecToResize, Args&... args);
    // shrinkContainer
    template<typename P1> void shrinkContainer(std::vector<P1>& vecToShrink);
    template<typename P1, typename ... Args> void shrinkContainer(std::vector<P1>& vecToShrink, Args&... args);
    template<typename P1> void shrinkContainer(std::vector<std::vector<P1>>& vecToShrink);
    template<typename P1, typename ... Args> void shrinkContainer(std::vector<std::vector<P1>>& vecToShrink, Args&... args);

    ArithmeticOperation() = delete;

    ArithmeticOperation(const std::string& name, const std::array<std::size_t, 5>& samples, const bool& maskStatus)
        : mOperationName{ name }, mSampleSizes{ samples }, mHasMask{ maskStatus }, OperationEventHandler{ *this, OperationTimeHandler }{}
   
public:

    // Getters for outside class operations
    const bool& getValidationStatus() const;
    const bool& getMaskStatus() const;
    const std::size_t& getCurrSampleSize() const;
    const std::size_t& getOpSampleSize(const int& option) const;
    const std::string& getOpName() const;

    // Launch ArithmeticOperation
    void startOpSeq(const int& userInput);
};

/* Templates for each possible container used by children of ArithmeticOperation
*  Template functions won't (can't) be inherited:
*  - children re-declare/define below functions (pertinent to operation)
*  - however, we re-use parent code by calling the parent function inside the child function
*/

// populateContainer
// Base case for 1D vector
template<typename P1>
void ArithmeticOperation::populateContainer(std::vector<P1>& vecToPop)
{
    if (vecToPop.size() > MaskAttributes::maskDim)
    {
        // Create local distribution on stack
        std::uniform_int_distribution randNum{ RandNumGen::minRand, RandNumGen::maxRand };

        // Generate random numbers via Lambda C++11 function, and place into vector
        generate(vecToPop.begin(), vecToPop.end(), [&randNum]() { return randNum(RandNumGen::mersenne); });
    }
    else // If we're passed a mask vector
    {
        // Create local distribution on stack
        std::uniform_int_distribution randNum{ RandNumGen::minMaskRand, RandNumGen::maxMaskRand };

        // Generate random numbers via Lambda C++11 function, and place into vector
        generate(vecToPop.begin(), vecToPop.end(), [&randNum]() { return randNum(RandNumGen::mersenne); });
    }
}
// Recursive case for 1D vector
template<typename P1, typename ... Args> 
void ArithmeticOperation::populateContainer(std::vector<P1>& vecToPop, Args&... args)
{
    this->populateContainer(vecToPop);
    this->populateContainer(args...);
}
// Base case for 2D vector
template<typename P1> 
void ArithmeticOperation::populateContainer(std::vector<std::vector<P1>>& vecToPop)
{
    // Create local distribution on stack
    std::uniform_int_distribution randNum{ RandNumGen::minRand, RandNumGen::maxRand };

    // Loop to populate 2D vector vecToPop
    // For each row
    for (auto rowIn{ 0 }; rowIn < vecToPop.size(); ++rowIn)
    {
        // For each column in that row
        for (auto colIn{ 0 }; colIn < vecToPop[rowIn].size(); ++colIn)
        {
            // Assign random number to vector of vector of ints to columns colIn of rows iRows
            vecToPop[rowIn][colIn] = randNum(RandNumGen::mersenne);
        }
    }
}
// Recursive case for 2D vector
template<typename P1, typename ... Args> 
void ArithmeticOperation::populateContainer(std::vector<std::vector<P1>>& vecToPop, Args&... args)
{
    this->populateContainer(vecToPop);
    this->populateContainer(args...);
}

// resizeContainer
// Base case for 1D vector
template<typename P1> 
void ArithmeticOperation::resizeContainer(const P1& newSize, std::vector<P1>& vecToResize)
{
    vecToResize.resize(newSize);
}
// Recursive case for 1D vector
template<typename P1, typename ... Args> 
void ArithmeticOperation::resizeContainer(const P1& newSize, std::vector<P1>& vecToResize, Args&... args)
{
    this->resizeContainer(newSize, vecToResize);
    this->resizeContainer(newSize, args...);
}
// Base case for 2D vector
template<typename P1> 
void ArithmeticOperation::resizeContainer(const P1& newSize, std::vector<std::vector<P1>>& vecToResize)
{
    vecToResize.resize(newSize, std::vector <P1>(2, 0));
}
// Recursive case for 2D vector
template<typename P1, typename ... Args> 
void ArithmeticOperation::resizeContainer(const P1& newSize, std::vector<std::vector<P1>>& vecToResize, Args&... args)
{
    this->resizeContainer(newSize, vecToResize);
    this->resizeContainer(newSize, args...);
}

// shrinkContainer
// Base case for 1D vector
template<typename P1> 
void ArithmeticOperation::shrinkContainer(std::vector<P1>& vecToShrink)
{
    vecToShrink.shrink_to_fit();
}
// Recursive case for 1D vector
template<typename P1, typename ... Args> 
void ArithmeticOperation::shrinkContainer(std::vector<P1>& vecToShrink, Args&... args)
{
    this->shrinkContainer(vecToShrink);
    this->shrinkContainer(args...);
}
// Base case for 2D vector
template<typename P1> 
void ArithmeticOperation::shrinkContainer(std::vector<std::vector<P1>>& vecToShrink)
{
    vecToShrink.shrink_to_fit();
}
// Recursive case for 2D vector
template<typename P1, typename ... Args> 
void ArithmeticOperation::shrinkContainer(std::vector<std::vector<P1>>& vecToShrink, Args&... args)
{
    this->shrinkContainer(vecToShrink);
    this->shrinkContainer(args...);
}
#endif