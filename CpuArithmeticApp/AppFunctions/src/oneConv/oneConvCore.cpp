#include "../../inc/oneConv/oneConvCore.h"

void oneConvCore()
{

    // Assign variable conSize with a user selected value
    oneConvConSize conSize = oneConvConSet(conSize);
    // Number of elements in the convolution mask
    int maskSize = 7;

    // Allocate main vector with size conSize
    std::vector<int> mainVec(conSize);
    // Allocate mask vector with m
    std::vector<int> maskVec(maskSize);

    // Popluate main vector and mask vector
    oneConvNumGen(mainVec, maskVec);

    // Start clock
    clock_t opStart = clock();

    // Start 1D Convolution operation
    oneConvFunc(mainVec, maskVec, conSize, maskSize);

    // Stop clock
    clock_t opEnd = clock();

    oneConvCheck(mainVec, maskVec, conSize, maskSize);

    // Calculate overall time spent to complete operation
    double completionTime = (opEnd - opStart) / (double)CLOCKS_PER_SEC;

    // Output timing to complete operation and container size
    std::cout << completionTime << "s 1D Convolution computation time, with a container size of " << conSize << ".\n\n";
    std::cout << "Returning to selection screen.\n\n";

    std::cout << "#########################################################################\n" <<
                 "#########################################################################\n" <<
                 "#########################################################################\n\n";
}