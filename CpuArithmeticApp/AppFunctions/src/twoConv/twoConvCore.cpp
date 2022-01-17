#include "../../inc/twoConv/twoConvCore.h"

constexpr int maskTwoDim { 7 };

void twoConvCore()
{
    // Assign variable conSize with a user selected value
    int conSize { twoConvConSet(conSize) };

    // Assign vectors mainVec(input vector) and resVec (resultant vector) a container size of conSize
    // mainVec is a matrix, therefore must be a 2D vector
    std::vector<int> mainVec(conSize * conSize), resVec(conSize * conSize, 0);

    // Assign 2D vector mask vector (maskVec) a container size of MASK_DIM * MASK_DIM
    // NOTE: ensure conSize * conSize / MASK_DIM * MASK_DIM  persists when moving onto 2D vectors, this ensures func and check work
    std::vector<int> maskVec(maskTwoDim * maskTwoDim);
   
    // Populate mainVec and maskVec
    std::cout << "\n2D Convolution: Populating main vector.\n";
    twoConvNumGen(mainVec);
    std::cout << "\n2D Convolution: Populating mask vector.\n";
    twoConvNumGen(maskVec);

    clock_t opStart { clock() };

    twoConvFunc(mainVec, maskVec, resVec, conSize, maskTwoDim);

    clock_t opEnd { clock() };

    twoConvCheck(mainVec, maskVec, resVec, conSize, maskTwoDim);

    // Calculate overall time spent to complete operation
    double completionTime{ ((static_cast<double>(opEnd)) - (static_cast<double>(opStart))) / (double)CLOCKS_PER_SEC };

    // Output timing to complete operation and container size
    std::cout << completionTime << "s 2D Convolution computation time, with a container size of " << conSize * conSize << ".\n\n";
    std::cout << "Returning to selection screen.\n\n";

    std::cout << "#########################################################################\n" <<
                 "#########################################################################\n" <<
                 "#########################################################################\n\n";
}
