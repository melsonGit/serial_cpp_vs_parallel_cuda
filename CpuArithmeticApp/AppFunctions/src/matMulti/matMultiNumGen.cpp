#include "../../inc/matMulti/matMultiNumGen.h"

void matMultiNumGen(std::vector<int>& a, std::vector<int>& b)
{
    // Re-seed rand() function for each run
    srand((uint16_t)time(NULL));

    // Generate random numbers via Lambda C++11 function, and place into vector
    std::generate(a.begin(), a.end(), []() { return rand() % 100; });
    std::generate(b.begin(), b.end(), []() { return rand() % 100; });
}