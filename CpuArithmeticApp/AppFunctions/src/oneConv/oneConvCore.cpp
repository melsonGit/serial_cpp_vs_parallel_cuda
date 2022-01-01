#include "../../inc/oneConv/oneConvCore.h"

#ifndef MASK_ONE_DIM
// Number of elements in the convolution mask
#define MASK_ONE_DIM 7
#endif

void oneConvCore()
{

    // Assign variable conSize with a user selected value
    oneConvConSize conSize { oneConvConSet(conSize) };

    // Allocate main vector and resultant vector with size conSize
    std::vector<int> mainVec(conSize), resVec(conSize);
    // Allocate mask vector with maskSize
    std::vector<int> maskVec(MASK_ONE_DIM);

    // Popluate main vector and mask vector
    oneConvNumGen(mainVec, maskVec);

    // Start clock
    clock_t opStart { clock() };

    // Start 1D Convolution operation
    oneConvFunc(mainVec, maskVec, resVec, conSize);

    // Stop clock
    clock_t opEnd { clock() };

    oneConvCheck(mainVec, maskVec, resVec, conSize);

    // Calculate overall time spent to complete operation
    double completionTime { (opEnd - opStart) / (double)CLOCKS_PER_SEC };

    // Output timing to complete operation and container size
    std::cout << completionTime << "s 1D Convolution computation time, with a container size of " << conSize << ".\n\n";
    std::cout << "Returning to selection screen.\n\n";

    std::cout << "#########################################################################\n" <<
                 "#########################################################################\n" <<
                 "#########################################################################\n\n";
}