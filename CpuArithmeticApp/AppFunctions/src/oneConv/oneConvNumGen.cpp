#include "../../inc/oneConv/oneConvNumGen.h"

void oneConvNumGen(std::vector<int>& vecToPop)
{
    // Re-seed rand() function for each run
    srand((unsigned int)time(NULL));

    // Generate random numbers via Lambda C++11 function, and place into vector
    generate(begin(vecToPop), end(vecToPop), []() { return rand() % 100; });
}