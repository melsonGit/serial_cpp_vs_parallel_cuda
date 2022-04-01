#include "../../inc/oneConv/oneConvCore.h"

using Clock = std::chrono::steady_clock;

void oneConvCore()
{

    // Assign variable conSize with a user selected value
    int conSize { oneConvConSet(conSize) };

    // Allocate main vector and resultant vector with size conSize
    std::vector<int> mainVec(conSize), resVec(conSize);
    // Allocate mask vector with maskSize
    std::vector<int> maskVec(maskAttributes::maskDim);

    // Popluate main vector and mask vector
    std::cout << "\n1D Convolution: Populating main vector.\n";
    oneConvNumGen(mainVec);
    std::cout << "\n1D Convolution: Populating mask vector.\n";
    oneConvNumGen(maskVec);

    // Start clock
    auto opStart { Clock::now() };

    // Start 1D Convolution operation
    oneConvFunc(mainVec, maskVec, resVec, conSize);

    // Stop clock
    auto opEnd { Clock::now() };

    oneConvCheck(mainVec, maskVec, resVec, conSize);

    // Output timing to complete operation and container size
    std::cout << "CPU 1D Convolution computation time (container size: " << conSize << "):\n"
              << std::chrono::duration_cast<std::chrono::microseconds>(opEnd - opStart).count() << " us\n"
              << std::chrono::duration_cast<std::chrono::milliseconds>(opEnd - opStart).count() << " ms\n\n"
              << "Returning to selection screen.\n\n"

              << "#########################################################################\n" <<
                 "#########################################################################\n" <<
                 "#########################################################################\n\n";
}