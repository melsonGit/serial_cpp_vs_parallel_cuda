// Sequential Vector Addition Program

#include <iostream>
#include <algorithm>
#include <vector>
#include "conSet.h"
#include "addFunc.h"
#include "numGen.h"

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

	// checkAdd(); func to determine if the output vector is indeed correct, print SUCCESS if correct or FAILED if not

	double diffs = (end - start) / (double)CLOCKS_PER_SEC;
	std::cout << diffs << "s Vector Addition computation time, with an container size of " << conSize << ".\n";
	std::cout << "SEQUENTIAL VECTOR ADDITION COMPUTATION SUCCESSFUL.\nShutting down program....\n";

	return EXIT_SUCCESS;

}
