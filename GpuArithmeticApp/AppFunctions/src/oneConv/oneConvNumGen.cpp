#include "../../inc/oneConv/oneConvNumGen.h"

void oneConvNumGen(std::vector<int>& vecToPop)
{
    // Re-seed rand() function for each run
    srand((unsigned int)time(NULL));

    // Generate random numbers and place into vector
    generate(begin(vecToPop), end(vecToPop), []() { return rand() % 100; });
}