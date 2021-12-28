#include "../../inc/oneConv/oneConvNumGen.h"

void oneConvNumGen(std::vector<int>& mainVec, std::vector<int>& maskVec)
{

    std::cout << "\n1D Convolution: Populating main and mask vectors.\n";

    // Re-seed rand() function for each run
    srand((unsigned int)time(NULL));

    // Generate random numbers via Lambda C++11 function, and place into vector
    generate(begin(mainVec), end(mainVec), []() { return rand() % 100; });
    // initialise mask || m mumber of elements in vector are randomised between 1 - 10
    generate(begin(maskVec), end(maskVec), []() { return rand() % 10; });

    std::cout << "\n1D Convolution: Populating complete.\n";
}