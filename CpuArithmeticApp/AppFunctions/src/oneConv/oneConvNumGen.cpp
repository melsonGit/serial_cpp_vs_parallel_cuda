#include "../../inc/oneConv/oneConvNumGen.h"

void oneConvNumGen(std::vector<int>& vecToPop)
{
	// Create local distribution on stack
	std::uniform_int_distribution randNum { randNumGen::minRand, randNumGen::maxRand };

	// Generate random numbers via Lambda C++11 function, and place into vector
	generate(vecToPop.begin(), vecToPop.end(), [&randNum]() { return randNum(randNumGen::mersenne); });
}