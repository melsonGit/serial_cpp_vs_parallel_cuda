#include "../../inc/vecAdd/vecAddCore.h"

using Clock = std::chrono::steady_clock;

void vecAddCore()
{
	// Assign variable conSize with a user selected value
	int conSize { vecAddConSet(conSize) };

	// Assign input vectors (inputVecA & inputVecB) and the output vector (resultVec) a container size of conSize
	std::vector<int> inputVecA(conSize), inputVecB(conSize), resultVec(conSize);

	// Populate vectors
	std::cout << "\nVector Addition: Populating 1 of 2 input vectors.\n";
	vecAddNumGen(inputVecA);
	std::cout << "\nVector Addition: Populating 2 of 2 input vectors.\n";
	vecAddNumGen(inputVecB);

	// Start clock
	auto opStart { Clock::now() };

	// Begin sequential vector addition operation
	vecAddFunc(inputVecA, inputVecB, resultVec);

	// Stop clock
	auto opEnd { Clock::now() };

	// Check output vector contents
	vecAddCheck(inputVecA, inputVecB, resultVec, conSize);

	// Output timing to complete operation and container size
	std::cout << std::chrono::duration_cast<std::chrono::milliseconds>(opEnd - opStart).count() 
			  << "ms Vector Addition computation time, with a container size of " << conSize << ".\n\n";
	std::cout << "Returning to selection screen.\n\n";

	std::cout << "#########################################################################\n" <<
				 "#########################################################################\n" <<
				 "#########################################################################\n\n";
}