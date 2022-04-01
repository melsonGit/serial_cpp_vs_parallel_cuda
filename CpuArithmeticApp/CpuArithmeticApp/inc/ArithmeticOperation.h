#pragma once
#ifndef ARITHMETIC_OPERATION
#define ARITHMETIC_OPERATION

#include "randNumGen.h"

#include <algorithm>
#include <array>
#include <random>
#include <ctime>
#include <string>
#include <string_view>
#include <vector> 

class ArithmeticOperation
{
protected:

    const std::string operationName{};
    const std::array<int, 5> sampleSizes{};

    ArithmeticOperation(const std::string& name, const std::array<int, 5>& samples) 
        : operationName{ name }, sampleSizes{ samples } {}

    // Operation-specific functions (.... where a template doesn't feel like an appropriate solution)
    virtual void startOperationSequence() = 0; // encompasses all operations inside it
    virtual void setContainer() = 0; // ask user - virtual as all displays to user will be different  
    virtual void launchOperation() = 0; // once populated, execute operation - virtual as all operations will be different (zero code redundancy)
    virtual void validateResults() = 0; // valid results of launchOperation() - virtual as all validation methods will be different (zero code redundancy)

    // Functions used by all operations
    template<typename P1> void populateContainer (std::vector<P1>&);
    template<typename P1> void populateContainer (std::vector<std::vector<P1>>&);// populate input containers with random numbers - template as there may be similar containers (code redundancy)
    //void recordResults(); // If valid, output to csv file

    // virtual ~ArithmeticOperation() = default; implement if we start to use pointers

public:

    const std::string_view getOperationName() const; // return operation name
};

// Templates for each possible container used by children of ArithmeticOperation
// Please update when adding a new operation

// Base case for 1D vector
template<typename P1>
void ArithmeticOperation::populateContainer(std::vector<P1>& vecToPop)
{
    // Create local distribution on stack
    std::uniform_int_distribution randNum{ randNumGen::minRand, randNumGen::maxRand };

    // Generate random numbers via Lambda C++11 function, and place into vector
    generate(vecToPop.begin(), vecToPop.end(), [&randNum]() { return randNum(randNumGen::mersenne); });
}

// Recursive case for 1D vector
template<typename P1, typename ... Args> 
void populateContainer(std::vector<P1>& vecToPop, Args&... args)
{
    populateContainer(vecToPop);
    populateContainer(args...);
}

// Base case for 2D vector
template<typename P1> 
void populateContainer(std::vector<std::vector<P1>>& vecToPop)
{
    // Create local distribution on stack
    std::uniform_int_distribution randNum{ randNumGen::minRand, randNumGen::maxRand };

    // Loop to populate 2D vector vecToPop
    // For each row
    for (auto iRow{ 0 }; iRow < vecToPop.size(); ++iRow)
    {
        // For each column in that row
        for (auto iCol{ 0 }; iCol < vecToPop[iRow].size(); ++iCol)
        {
            // Assign random number to vector of vector of ints to columns iCol of rows iRows
            vecToPop[iRow][iCol] = randNum(randNumGen::mersenne);
        }
    }
}

// Recursive case for 2D vector
template<typename P1, typename ... Args> 
void populateContainer(std::vector<std::vector<P1>>& vecToPop, Args&... args)
{
    populateContainer(vecToPop);
    populateContainer(args...);
}
#endif