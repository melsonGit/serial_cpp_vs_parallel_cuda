#include "../../inc/vecAdd/vecAddCore.h"

void vecAddCore()
{
	// Assign variable conSize with a user selected value
	int conSize { vecAddConSet(conSize) };

	// Assign input vectors (a & b) and the output vector (c) a container size of conSize
	std::vector<int> inputVecA(conSize), inputVecB(conSize), resultVec(conSize);

	// Populate vectors
	std::cout << "\nVector Addition: Populating 1 of 2 input vectors.\n";
	vecAddNumGen(inputVecA);
	std::cout << "\nVector Addition: Populating 2 of 2 input vectors.\n";
	vecAddNumGen(inputVecB);

	// Start clock
	clock_t opStart { clock() };

	// Begin sequential vector addition operation
	vecAddFunc(inputVecA, inputVecB, resultVec);

	// Stop clock
	clock_t opEnd { clock() };

	// Check output vector contents
	vecAddCheck(inputVecA, inputVecB, resultVec, conSize);

	// Calculate overall time spent to complete operation
	double completionTime { ((static_cast<double>(opEnd)) - (static_cast<double>(opStart))) / (double)CLOCKS_PER_SEC };

	// Output timing to complete operation and container size
	std::cout << completionTime << "s Vector Addition computation time, with a container size of " << conSize << ".\n\n";
	std::cout << "Returning to selection screen.\n\n";

	std::cout << "#########################################################################\n" <<
				 "#########################################################################\n" <<
				 "#########################################################################\n\n";
}