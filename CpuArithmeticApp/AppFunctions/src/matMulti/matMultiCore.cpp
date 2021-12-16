#include "../../inc/matMulti/matMultiCore.h"

void matMultiCore()
{

	// Assign variable conSize with a user selected value
	matMultiConSize conSize = matMultiConSet(conSize);

	// Assign input vectors (a & b) and the output vector (c) a container size of conSize
	std::vector<int> a(conSize * conSize), b(conSize * conSize), c(conSize * conSize);

	// Populate vectors
	matMultiNumGen(a, b);

	// Start clock
	clock_t opStart = clock();

	// Begin sequential vector addition operation
	matMultiFunc(a, b, c, conSize);

	// Stop clock
	clock_t opEnd = clock();

	// Check output vector contents
	matMultiCheck(a, b, c);

	// Calculate overall time spent to complete operation
	double completionTime = (opEnd - opStart) / (double)CLOCKS_PER_SEC;

	// Output timing to complete operation and container size
	std::cout << completionTime << "s Matrix Multiplication computation time, with a container size of " << conSize << ".\n\n";
	std::cout << "Returning to selection screen.\n\n";
}