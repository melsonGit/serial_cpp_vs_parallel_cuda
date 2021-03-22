// Vector Addition - Serial
// Sources: 
// https://solarianprogrammer.com/2012/04/11/vector-addition-benchmark-c-cpp-fortran/
// https://github.com/CoffeeBeforeArch/cuda_programming/blob/master/vectorAdd/baseline/vectorAdd.cu

#include <iostream>
#include <algorithm>
#include <vector>


using namespace std;

// Function Prototypes
size_t element_set(size_t);
void add(const vector<unsigned int>, const vector<unsigned int>, const vector<unsigned int>);

int main()
{
	// Call element_set function to assign variable no_elements with a user selected value
	static size_t no_elements = element_set(no_elements);

	vector<unsigned int> a(no_elements), b(no_elements), c(no_elements);

	for (size_t i = 0; i < no_elements; ++i) {
		a[i] = 1 / (unsigned int)(i + 1);
		b[i] = a[i];
	} 

	clock_t start = clock();

	add(a, b, c);

	clock_t end = clock();
	double diffs = (end - start) / (double)CLOCKS_PER_SEC;
	cout << diffs << "s Vector Addition computation time, with an element size of " << no_elements << ".\n";
	cout << "SEQUENTIAL VECTOR ADDITION COMPUTATION SUCCESSFUL.\nShutting down program....\n";

	return EXIT_SUCCESS;

}

// Function Declarations
size_t element_set(size_t element_size) {

	int temp_input;

	cout << "Please select vector addition element sample size from the options below:\n";
	cout << "1. 9,000\n";
	cout << "2. 90,000\n";
	cout << "3. 9,000,00\n";
	cout << "4. 9,000,000\n";
	cout << "5. 65,000,000\n";
	cin >> temp_input;

	if (temp_input <= 0 || temp_input >= 6)
	{
		cout << "\n\nNo correct option selected!\nShutting down program....\n";
		return EXIT_FAILURE;
	}

	if (temp_input == 1) {
		element_size = 9000;
	}
	else if (temp_input == 2) {
		element_size = 90000;
	}
	else if (temp_input == 3) {
		element_size = 900000;
	}
	else if (temp_input == 4) {
		element_size = 9000000;
	}
	else if (temp_input == 5) {
		element_size = 65000000;
	}

	return element_size;
}
void add(const vector<unsigned int> a, const vector<unsigned int> b, vector<unsigned int> c) {
	// Lambda Transform c++11
	transform(a.begin(), a.end(), b.begin(), c.begin(), 
		[](double a, double b) {return a + b; });
}