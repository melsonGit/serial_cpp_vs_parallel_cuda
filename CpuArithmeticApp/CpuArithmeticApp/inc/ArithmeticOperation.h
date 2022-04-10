#pragma once
#ifndef ARITHMETIC_OPERATION
#define ARITHMETIC_OPERATION

#include "randNumGen.h"

#include <algorithm>
#include <array>
#include <ctime>
#include <random>
#include <string>
#include <string_view>

class ArithmeticOperation
{
protected:

    const std::string mOperationName{};
    const std::array<std::size_t, 5> mSampleSizes{};

    ArithmeticOperation(const std::string& name, const std::array<std::size_t, 5>& samples)
        : mOperationName{ name }, mSampleSizes{ samples } {}

    // virtual ~ArithmeticOperation() = default; implement if we start to use pointers
    
    // Operation-specific functions (.... where a template doesn't feel like an appropriate solution)
    virtual void setContainer(const int& userInput) = 0;
    virtual void launchOp() = 0;
    virtual void validateResults() = 0; // valid results of launchOp()
    //void recordResults(); // If valid, output to csv file

    // Functions used by all operations - template as there may be similar containers
    template<typename P1> void populateContainer (std::vector<P1>& vecToPop);
    template<typename P1, typename ... Args> void populateContainer(std::vector<P1>& vecToPop, Args&... args);
    template<typename P1> void populateContainer (std::vector<std::vector<P1>>& vecToPop);
    template<typename P1, typename ... Args> void populateContainer(std::vector<std::vector<P1>>& vecToPop, Args&... args);
    
public:

    void startOpSeq(const int& userInput);

    const std::string_view getOpName() const;
    const std::size_t getOpSampleSize(const int& option) const;
};

/* Templates for each possible container used by children of ArithmeticOperation - Please update when a new container is required
*  Template functions won't (can't) be inherited:
*  - children re-declare/define below functions (pertinent to operation)
*  - however, we re-use parent code by calling the parent function inside the child function
*/

using namespace randNumGen;

// Base case for 1D vector
template<typename P1>
void ArithmeticOperation::populateContainer(std::vector<P1>& vecToPop)
{
    // Create local distribution on stack
    std::uniform_int_distribution randNum{ minRand, maxRand };

    // Generate random numbers via Lambda C++11 function, and place into vector
    generate(vecToPop.begin(), vecToPop.end(), [&randNum]() { return randNum(mersenne); });
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
void populateContainer(std::vector<std::vector<P1>>& vecToPop)
{
    // Create local distribution on stack
    std::uniform_int_distribution randNum{ minRand, maxRand };

    // Loop to populate 2D vector vecToPop
    // For each row
    for (auto iRow{ 0 }; iRow < vecToPop.size(); ++iRow)
    {
        // For each column in that row
        for (auto iCol{ 0 }; iCol < vecToPop[iRow].size(); ++iCol)
        {
            // Assign random number to vector of vector of ints to columns iCol of rows iRows
            vecToPop[iRow][iCol] = randNum(mersenne);
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