// Sequential Vector Addition Program


#include <iostream>
#include <algorithm>
#include <vector>

using std::cout;
using std::cin;
using std::generate;
using std::vector;

// Function Prototypes
int element_set(int);
void add(vector<int>, vector<int>, vector<int>);

int main()
{
	// Call element_set function to assign variable no_elements with a user selected value
	static int no_elements = element_set(no_elements);

	vector<int> a(no_elements), b(no_elements), c(no_elements);

    clock_t start = clock();

	// Generate random numbers via Lambda C++11 function, and place into vector
	generate(a.begin(), a.end(), []() { return rand() % 100; });
	generate(b.begin(), b.end(), []() { return rand() % 100; });

	add(a, b, c);

	clock_t end = clock();

	double diffs = (end - start) / (double)CLOCKS_PER_SEC;
	cout << diffs << "s Vector Addition computation time, with an element size of " << no_elements << ".\n";
	cout << "SEQUENTIAL VECTOR ADDITION COMPUTATION SUCCESSFUL.\nShutting down program....\n";

	return EXIT_SUCCESS;

}

// Function Declarations
int element_set(int element_size) {

    int temp_input;

    cout << "Please select vector addition element sample size from the options below:\n";
    cout << "1. 25,000,000\n";
    cout << "2. 35,000,000\n";
    cout << "3. 45,000,000\n";
    cout << "4. 55,000,000\n";
    cout << "5. 65,000,000\n";
    cin >> temp_input;

    if (temp_input <= 0 || temp_input >= 6)
    {
        cout << "\n\nNo correct option selected!\nShutting down program....\n";
        return EXIT_FAILURE;
    }
      // 25 million elements
    if (temp_input == 1) {
        element_size = 25000000;
    } // 35 million elements
    else if (temp_input == 2) {
        element_size = 35000000;
    } // 45 million elements
    else if (temp_input == 3) {
        element_size = 45000000;
    } // 55 million elements
    else if (temp_input == 4) {
        element_size = 55000000;
    } // 65 million elements
    else if (temp_input == 5) {
        element_size = 65000000;
    }

    return element_size;
}
void add(vector<int> a, vector<int> b, vector<int> c) {
	// Add contents from vector 'a' and 'b' into vector 'c' || Transform using a Lambda C++11
	transform(a.begin(), a.end(), b.begin(), c.begin(), 
		[](int a, int b) {return a + b; });
}

