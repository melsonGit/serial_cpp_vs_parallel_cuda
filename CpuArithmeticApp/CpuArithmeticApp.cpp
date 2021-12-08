#include <iostream>
#include <algorithm>
#include <vector>
#include "../CpuArithmeticApp/AppFunctions/inc/allHeaders.h"

int main()
{
	// Assign variable conSize with a user selected value
	int conSize = conSet(conSize);

	// Assign input vectors (a & b) and the output vector (c) a container size of conSize
	std::vector<int> a(conSize), b(conSize), c(conSize);

	// Populate vectors
	numGen(a, b);

	// Start clock
	clock_t start = clock();

	// Begin sequential vector addition operation
	add(a, b, c);

	// Stop clock
	clock_t end = clock();

	// Check output vector contents
	checkAdd(a, b, c);

	double diffs = (end - start) / (double)CLOCKS_PER_SEC;
	std::cout << diffs << "s Vector Addition computation time, with a container size of " << conSize << ".\n";
	std::cout << "Closing program...\n";

	return EXIT_SUCCESS;

}
