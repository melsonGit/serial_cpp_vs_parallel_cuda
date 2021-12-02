// Sequential Vector Addition Program

#include <iostream>
#include <algorithm>
#include <vector>
#include "popVec.h"
#include "add.h"


using std::cout;
using std::cin;
using std::generate;
using std::vector;

// Function Prototypes
void add(vector<int>, vector<int>, vector<int>);

int main()
{
	// Call elementSet function to assign variable noElements with a user selected value
	popVec noElements{0};

	// Allocate vectors a, b and c the size of noElements
	vector<int> a(noElements), b(noElements), c(noElements);

    clock_t start = clock();

	// Generate random numbers via Lambda C++11 function, and place into vector
	generate(a.begin(), a.end(), []() { return rand() % 100; });
	generate(b.begin(), b.end(), []() { return rand() % 100; });

	add(a, b, c);

	clock_t end = clock();

	double diffs = (end - start) / (double)CLOCKS_PER_SEC;
	cout << diffs << "s Vector Addition computation time, with an element size of " << noElements << ".\n";
	cout << "SEQUENTIAL VECTOR ADDITION COMPUTATION SUCCESSFUL.\nShutting down program....\n";

	return EXIT_SUCCESS;

}


void add(vector<int> a, vector<int> b, vector<int> c) {
	// Add contents from vector 'a' and 'b' into vector 'c' || Transform using a Lambda C++11
	transform(a.begin(), a.end(), b.begin(), c.begin(), 
		[](int a, int b) {return a + b; });
}

