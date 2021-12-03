// Sequential Vector Addition Program

#include <iostream>
#include <algorithm>
#include <vector>
#include "conSet.h"
#include "addFunc.h"

int main()
{
	// Call conSet function to assign variable conSize with a user selected value
	int conSize = conSet(conSize);
	// Assign input vectors (a & b) and the output vector (c) a container size conSize
	std::vector<int> a(conSize), b(conSize), c(conSize);

    clock_t start = clock();

	// Re-seed rand() function for each run
	srand((unsigned int)time(NULL));

	// Generate random numbers via Lambda C++11 function, and place into vector
	std::generate(a.begin(), a.end(), []() { return rand() % 100; });
	std::generate(b.begin(), b.end(), []() { return rand() % 100; });

	add(a, b, c);

	clock_t end = clock();

	double diffs = (end - start) / (double)CLOCKS_PER_SEC;
	std::cout << diffs << "s Vector Addition computation time, with an container size of " << conSize << ".\n";
	std::cout << "SEQUENTIAL VECTOR ADDITION COMPUTATION SUCCESSFUL.\nShutting down program....\n";

	return EXIT_SUCCESS;

}
