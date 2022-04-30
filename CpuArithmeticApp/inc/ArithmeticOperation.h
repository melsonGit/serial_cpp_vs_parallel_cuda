#pragma once
#ifndef ARITHMETIC_OPERATION
#define ARITHMETIC_OPERATION

#include "OperationEventHandler.h"
#include "OperationTimer.h"
#include "randNumGen.h"
#include "MaskAttributes.h"

#include <algorithm>
#include <array>
#include <ctime>
#include <random>
#include <string>

class ArithmeticOperation
{
protected:

    // Characteristics unique to every ArithmeticOperation child
    const std::string mOperationName{};
    const std::array<std::size_t, 5> mSampleSizes{};
    const bool hasMask{};

    // Var checks for ArithmeticOperation function checks 
    bool hasPassedValidation{};
    std::size_t currSampleSize{};
    int vecIndex{ 99 }; // Default first run value (see setContainer() of any child). Any number outside 0 - 6 is fine but just to be safe

    // Operation Tools
    OperationEventHandler OperationEventHandler;
    OperationTimer OperationTimer{};
    void recordResults();
    void bestOperationTimes();
    
    ArithmeticOperation(const std::string& name, const std::array<std::size_t, 5>& samples, const bool& maskStatus)
        : mOperationName{ name }, mSampleSizes{ samples }, hasMask{ maskStatus }, OperationEventHandler{*this, OperationTimer} {}

    // Operation-specific functions (.... where a template doesn't feel like an appropriate solution (for now))
    virtual void setContainer(const int& userInput) = 0;
    virtual void launchOp() = 0;
    virtual void validateResults() = 0;

    // Functions used by all operations
    template<typename P1> void populateContainer (std::vector<P1>& vecToPop);
    template<typename P1, typename ... Args> void populateContainer(std::vector<P1>& vecToPop, Args&... args);
    template<typename P1> void populateContainer (std::vector<std::vector<P1>>& vecToPop);
    template<typename P1, typename ... Args> void populateContainer(std::vector<std::vector<P1>>& vecToPop, Args&... args);
    
public:

    const bool getValidationStatus() const;
    const int getVecIndex() const;
    const std::size_t getCurrSampleSize() const;
    const bool getMaskStatus() const;
    const std::string getOpName() const;
    //const std::string_view viewOpName() const; may be of use when we just want to display our string
    const std::size_t getOpSampleSize(const int& option) const;
    
    void setCurrSampleSize(const int& index);
    void setValidationStatus(const bool& validationResult);
    void setVecIndex(const int& newIndex);
    void startOpSeq(const int& userInput);
};

/* Templates for each possible container used by children of ArithmeticOperation - Please update when a new container is required
*  Template functions won't (can't) be inherited:
*  - children re-declare/define below functions (pertinent to operation)
*  - however, we re-use parent code by calling the parent function inside the child function
*/

using namespace randNumGen;
using namespace MaskAttributes;

// Base case for 1D vector
template<typename P1>
void ArithmeticOperation::populateContainer(std::vector<P1>& vecToPop)
{
    if (vecToPop.size() > maskDim)
    {
        // Create local distribution on stack
        std::uniform_int_distribution randNum{ minRand, maxRand };

        // Generate random numbers via Lambda C++11 function, and place into vector
        generate(vecToPop.begin(), vecToPop.end(), [&randNum]() { return randNum(mersenne); });
    }
    else // If we're passed a mask vector
    {
        // Create local distribution on stack
        std::uniform_int_distribution randNum{ minMaskRand, maxMaskRand };

        // Generate random numbers via Lambda C++11 function, and place into vector
        generate(vecToPop.begin(), vecToPop.end(), [&randNum]() { return randNum(mersenne); });
    }
}

// Recursive case for 1D vector
template<typename P1, typename ... Args> 
void ArithmeticOperation::populateContainer(std::vector<P1>& vecToPop, Args&... args)
{
    populateContainer(vecToPop);
    populateContainer(args...);
}

// Base case for 2D vector
template<typename P1> 
void ArithmeticOperation::populateContainer(std::vector<std::vector<P1>>& vecToPop)
{
    // Create local distribution on stack
    std::uniform_int_distribution randNum{ minRand, maxRand };

    // Loop to populate 2D vector vecToPop
    // For each row
    for (auto rowIn{ 0 }; rowIn < vecToPop.size(); ++rowIn)
    {
        // For each column in that row
        for (auto colIn{ 0 }; colIn < vecToPop[rowIn].size(); ++colIn)
        {
            // Assign random number to vector of vector of ints to columns colIn of rows iRows
            vecToPop[rowIn][colIn] = randNum(mersenne);
        }
    }
}

// Recursive case for 2D vector
template<typename P1, typename ... Args> 
void ArithmeticOperation::populateContainer(std::vector<std::vector<P1>>& vecToPop, Args&... args)
{
    populateContainer(vecToPop);
    populateContainer(args...);
}
#endif
