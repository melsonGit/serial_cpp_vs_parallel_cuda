#include "../../inc/vecAdd/vecAddNumGen.h"

void vecAddNumGen(std::vector<int>& vecToPop)
{
	// Re-seed rand() function for each run
	srand((unsigned int)time(NULL));

	generate(vecToPop.begin(), vecToPop.end(), []() {return rand() % 100; });
}