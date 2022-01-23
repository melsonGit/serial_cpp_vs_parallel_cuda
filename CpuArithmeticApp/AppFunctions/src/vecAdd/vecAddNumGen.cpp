#include "../../inc/vecAdd/vecAddNumGen.h"

void vecAddNumGen(std::vector<int>& vecToPop)
{
	// Re-seed rand() function for each run
	srand((unsigned int)time(NULL));

	// Generate random numbers via Lambda C++11 function, and place into vector
	std::generate(vecToPop.begin(), vecToPop.end(), []() { return rand() % 100; });
}