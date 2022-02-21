#include "../../inc/twoConv/twoConvCore.h"

using Clock = std::chrono::steady_clock;

void twoConvCore()
{
    // Assign variable conSize with a user selected value
    int conSize { twoConvConSet(conSize) };

    // Assign vectors mainVec(input vector) and resVec (resultant vector) a container size of conSize
    // mainVec is a matrix, therefore must be a 2D vector
    std::vector<int> mainVec(conSize * conSize), resVec(conSize * conSize, 0);

    // Assign 2D vector mask vector (maskVec) a container size of MASK_DIM * MASK_DIM
    // NOTE: ensure conSize * conSize / MASK_DIM * MASK_DIM  persists when moving onto 2D vectors, this ensures func and check work
    std::vector<int> maskVec(maskAttributes::maskDim * maskAttributes::maskDim);
   
    // Populate mainVec and maskVec
    std::cout << "\n2D Convolution: Populating main vector.\n";
    twoConvNumGen(mainVec);
    std::cout << "\n2D Convolution: Populating mask vector.\n";
    twoConvNumGen(maskVec);

    // Start clock
    auto opStart { Clock::now() };

    twoConvFunc(mainVec, maskVec, resVec, conSize);

    // Stop clock
    auto opEnd { Clock::now() };

    twoConvCheck(mainVec, maskVec, resVec, conSize);

    // Output timing to complete operation and container size
    std::cout << "CPU 2D Convolution computation time (container size: " << conSize * conSize << "):\n"
              << std::chrono::duration_cast<std::chrono::microseconds>(opEnd - opStart).count() << " us\n"
              << std::chrono::duration_cast<std::chrono::milliseconds>(opEnd - opStart).count() << " ms\n\n"
              << "Returning to selection screen.\n\n"

              << "#########################################################################\n" <<
                 "#########################################################################\n" <<
                 "#########################################################################\n\n";
}