#include "../../inc/oneConv/oneConvCore.h"

constexpr int maskOneDim { 7 };

void oneConvCore()
{

    // Assign variable conSize with a user selected value
    int conSize { oneConvConSet(conSize) };

    // Allocate main vector and resultant vector with size conSize
    std::vector<int> mainVec(conSize), resVec(conSize);
    // Allocate mask vector with maskSize
    std::vector<int> maskVec(maskOneDim);

    // Popluate main vector and mask vector
    std::cout << "\n1D Convolution: Populating main vector.\n";
    oneConvNumGen(mainVec);
    std::cout << "\n1D Convolution: Populating mask vector.\n";
    oneConvNumGen(maskVec);

    // Start clock
    clock_t opStart { clock() };

    // Start 1D Convolution operation
    oneConvFunc(mainVec, maskVec, resVec, conSize, maskOneDim);

    // Stop clock
    clock_t opEnd { clock() };

    oneConvCheck(mainVec, maskVec, resVec, conSize, maskOneDim);

    // Calculate overall time spent to complete operation
    double completionTime{ ((static_cast<double>(opEnd)) - (static_cast<double>(opStart))) / (double)CLOCKS_PER_SEC };

    // Output timing to complete operation and container size
    std::cout << completionTime << "s 1D Convolution computation time, with a container size of " << conSize << ".\n\n";
    std::cout << "Returning to selection screen.\n\n";

    std::cout << "#########################################################################\n" <<
                 "#########################################################################\n" <<
                 "#########################################################################\n\n";
}