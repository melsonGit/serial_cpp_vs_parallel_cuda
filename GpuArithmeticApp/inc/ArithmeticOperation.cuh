#pragma once
#ifndef ARITHMETIC_OPERATION
#define ARITHMETIC_OPERATION

#include "OperationEventHandler.h"
#include "OperationTimeHandler.h"
#include "RandNumGen.h"
#include "MaskAttributes.h"

#include "cuda_runtime.h"
#include "device_launch_parameters.h"

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
    OperationTimeHandler OperationTimeHandler; // Collects and provides kernel execution time
    OperationEventHandler OperationEventHandler; // Reports operation events to interface

    // CUDA Specific Variables - Usage specifically to work with CUDA
    std::size_t mTemp2DConSize{}; // 2D containers require a pre-squared container value for use in kernel execution and result validation  
    std::size_t mMemSize{}; // Utilised in device variable memory allocation, required for copying of data to and from device (refer to child allocateMemToDevice())
    std::size_t m1DMaskMemSize{}; // As above, but for 1D mask containers
    std::size_t m2DMaskMemSize{}; // As above, but for 2D mask containers
    const std::size_t mTHREADS{ 32 }; // No. of threads to execute per mBLOCKS | 32 because warps come in sizes of 32 and must by default by divisible by 32 to work effectively
    std::size_t mBLOCKS{}; // Number of Blocks holding mTHREADS used to execute kernel simultaneously
    /*
    * dim3 CUDA types allow mTHREADSand mBLOCKS to work within 2D/3D containers
    * when initialised, we're given a variable capable of working in a row/column fashion when used as a kernel launch parameter
    * e.g. kernels are provided with mBLOCKS * mBLOCKS of mTHREADS * mTHREADS
    */  
    dim3 mDimThreads{}; // 2D mTHREADS launch parameter
    dim3 mDimBlocks{}; // 2D mBLOCKS launch parameter

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
    
    // CUDA Specific Functions
    virtual void allocateMemToDevice() = 0; // Memory is assigned to device variables as per use sample size selection
    virtual void copyHostToDevice() = 0; // Copy host containers to device for kernel operation.
    virtual void copyDeviceToHost() = 0; // Copy device containers back to host once kernel operation is complete.
    virtual void freeDeviceData() = 0; // Free device variable memory
    void update1DKernelVars();
    void update2DKernelVars();
    void updateDimStructs();
    void update1DMemSize();
    void update2DMemSize();
    void update1DMaskMemSize();
    void update2DMaskMemSize();
    void update1DBlockSize();
    void update2DBlockSize();

    // Setters & Getters
    void setCurrSampleSize(const int& index);
    void setValidationStatus(const bool& validationResult);
    void setVecIndex(const int& newIndex);
    const int& getVecIndex() const;
    const std::size_t& getTemp2DConSize() const;

    // Send events to OperationEventHandler
    void updateEventHandler(const EventDirectives& event);
    
    // Container functions used by all operations (1D & 2D containers listed respectively)
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
        : mOperationName{ name }, mSampleSizes{ samples }, mHasMask{ maskStatus }, OperationEventHandler{ *this, OperationTimeHandler } {}
   
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
        std::uniform_int_distribution<> randNum{ RandNumGen::minRand, RandNumGen::maxRand };

        // Generate random numbers via Lambda C++11 function, and place into vector
        generate(vecToPop.begin(), vecToPop.end(), [&randNum]() { return randNum(RandNumGen::mersenne); });
    }
    else // If we're passed a mask vector
    {
        // Create local distribution on stack
        std::uniform_int_distribution<> randNum{ RandNumGen::minMaskRand, RandNumGen::maxMaskRand };

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
    std::uniform_int_distribution<> randNum{ RandNumGen::minRand, RandNumGen::maxRand };

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