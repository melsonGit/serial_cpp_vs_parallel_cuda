#include "../../inc/twoConv/twoConvCore.h"

// 7 x 7 convolutional mask
#define MASK_TWO_DIM 7
// Amount the the matrix will hang over the matrix
#define MASK_OFFSET (MASK_TWO_DIM / 2)

void twoConvCore()
{
    // Assign variable conSize with a user selected value
    twoConvConSize conSize { twoConvConSet(conSize) };

    // Assign vectors mainVec(input vector) and resVec (resultant vector) a container size of conSize
    std::vector<int> mainVec(conSize), resVec(conSize);
    // Assign 2D vector maskVec(maskVec) a container size of MASK_DIM * MASK_DIM
    std::vector<std::vector<int>> maskVec(MASK_TWO_DIM, std::vector<int>(2, 0));

    // Populate mainVec and maskVec
    twoConvNumGen(mainVec, maskVec);

    clock_t opStart { clock() };

    twoConvFunc(mainVec, maskVec, resVec, conSize);

    clock_t opEnd { clock() };

    twoConvCheck(mainVec, maskVec, resVec, conSize);

    // Calculate overall time spent to complete operation
    double completionTime { (opEnd - opStart) / (double)CLOCKS_PER_SEC };

    // Output timing to complete operation and container size
    std::cout << completionTime << "s 2D Convolution computation time, with a container size of " << conSize << ".\n\n";
    std::cout << "Returning to selection screen.\n\n";

    std::cout << "#########################################################################\n" <<
                 "#########################################################################\n" <<
                 "#########################################################################\n\n";
}
