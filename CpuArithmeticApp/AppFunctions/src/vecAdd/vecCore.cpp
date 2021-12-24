#include "../../inc/vecAdd/vecCore.h"

void vecCore()
{

	// Assign variable conSize with a user selected value
	vecAddConSize conSize = vecConSet(conSize);

	// Assign input vectors (a & b) and the output vector (c) a container size of conSize
	std::vector<int> a(conSize), b(conSize), c(conSize);

	// Populate vectors
	vecNumGen(a, b);

	// Start clock
	clock_t opStart = clock();

	// Begin sequential vector addition operation
	vecAddFunc(a, b, c);

	// Stop clock
	clock_t opEnd = clock();

	// Check output vector contents
	vecCheck(a, b, c, conSize);

	// Calculate overall time spent to complete operation
	double completionTime = (opEnd - opStart) / (double)CLOCKS_PER_SEC;

	// Output timing to complete operation and container size
	std::cout << completionTime << "s Vector Addition computation time, with a container size of " << conSize << ".\n\n";
	std::cout << "Returning to selection screen.\n\n";

	std::cout << "#########################################################################\n" <<
				 "#########################################################################\n" <<
				 "#########################################################################\n\n";
}