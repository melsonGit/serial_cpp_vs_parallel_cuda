#include "../../inc/vecAdd/vecAddCore.h"

void vecAddCore()
{
	// Assign variable conSize with a user selected value
	int conSize { vecAddConSet(conSize) };

	// Assign input vectors (a & b) and the output vector (c) a container size of conSize
	std::vector<int> a(conSize), b(conSize), c(conSize);

	// Populate vectors
	std::cout << "\nVector Addition: Populating 1 of 2 input vectors.\n";
	vecAddNumGen(a);
	std::cout << "\nVector Addition: Populating 2 of 2 input vectors.\n";
	vecAddNumGen(b);

	// Start clock
	clock_t opStart { clock() };

	// Begin sequential vector addition operation
	vecAddFunc(a, b, c);

	// Stop clock
	clock_t opEnd { clock() };

	// Check output vector contents
	vecAddCheck(a, b, c, conSize);

	// Calculate overall time spent to complete operation
	double completionTime { (opEnd - opStart) / (double)CLOCKS_PER_SEC };

	// Output timing to complete operation and container size
	std::cout << completionTime << "s Vector Addition computation time, with a container size of " << conSize << ".\n\n";
	std::cout << "Returning to selection screen.\n\n";

	std::cout << "#########################################################################\n" <<
				 "#########################################################################\n" <<
				 "#########################################################################\n\n";
}