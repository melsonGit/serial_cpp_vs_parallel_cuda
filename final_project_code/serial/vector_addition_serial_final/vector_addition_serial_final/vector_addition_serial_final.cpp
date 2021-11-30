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
int elementSet(int);
void add(vector<int>, vector<int>, vector<int>);

int main()
{
	// Call elementSet function to assign variable noElements with a user selected value
	static int noElements = elementSet(noElements);

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

// Function Declarations
int elementSet(int elementSize) {

    int tempInput;

    cout << "Please select vector addition element sample size from the options below:\n";
    cout << "1. 25,000,000\n";
    cout << "2. 35,000,000\n";
    cout << "3. 45,000,000\n";
    cout << "4. 55,000,000\n";
    cout << "5. 65,000,000\n";
    cin >> tempInput;

    if (tempInput <= 0 || tempInput >= 6)
    {
        cout << "\n\nNo correct option selected!\nShutting down program....\n";
        return EXIT_FAILURE;
    }
      // 25 million elements
    if (tempInput == 1) {
        elementSize = 25000000;
    } // 35 million elements
    else if (tempInput == 2) {
        elementSize = 35000000;
    } // 45 million elements
    else if (tempInput == 3) {
        elementSize = 45000000;
    } // 55 million elements
    else if (tempInput == 4) {
        elementSize = 55000000;
    } // 65 million elements
    else if (tempInput == 5) {
        elementSize = 65000000;
    }

    return elementSize;
}
void add(vector<int> a, vector<int> b, vector<int> c) {
	// Add contents from vector 'a' and 'b' into vector 'c' || Transform using a Lambda C++11
	transform(a.begin(), a.end(), b.begin(), c.begin(), 
		[](int a, int b) {return a + b; });
}

