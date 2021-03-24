// Vector Addition - Serial
// Sources: 
// https://solarianprogrammer.com/2012/04/11/vector-addition-benchmark-c-cpp-fortran/
// https://github.com/CoffeeBeforeArch/cuda_programming/blob/master/vectorAdd/baseline/vectorAdd.cu
// https://thispointer.com/how-to-fill-a-vector-with-random-numbers-in-c/

#include <iostream>
#include <algorithm>
#include <vector>

// Function Prototypes
int element_set(int);
void add(const std::vector<int>, const std::vector<int>, const std::vector<int>);

int main()
{
	// Call element_set function to assign variable no_elements with a user selected value
	static int no_elements = element_set(no_elements);

	std::vector<int> a(no_elements), b(no_elements), c(no_elements);

	// Generate random numbers via Lambda C++11 function, and place into vector
	std::generate(a.begin(), a.end(), []() {
		return rand() % 100;
		});
	std::generate(b.begin(), b.end(), []() {
		return rand() % 100;
		});

	// Slower alternative non-random vector initialisation 
	//for (size_t i = 0; i < no_elements; ++i) {
	//	a[i] = 1 / (unsigned int)(i + 1);
	//	b[i] = a[i];
	//} 

	clock_t start = clock();

	add(a, b, c);

	clock_t end = clock();

	double diffs = (end - start) / (double)CLOCKS_PER_SEC;
	std::cout << diffs << "s Vector Addition computation time, with an element size of " << no_elements << ".\n";
	std::cout << "SEQUENTIAL VECTOR ADDITION COMPUTATION SUCCESSFUL.\nShutting down program....\n";

	return EXIT_SUCCESS;

}

// Function Declarations
int element_set(int element_size) {

	int temp_input;

	std::cout << "Please select vector addition element sample size from the options below:\n";
	std::cout << "1. 9,000\n";
	std::cout << "2. 90,000\n";
	std::cout << "3. 9,000,00\n";
	std::cout << "4. 9,000,000\n";
	std::cout << "5. 65,000,000\n";
	std::cin >> temp_input;

	if (temp_input <= 0 || temp_input >= 6)
	{
		std::cout << "\n\nNo correct option selected!\nShutting down program....\n";
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
void add(const std::vector<int> a, const std::vector<int> b, std::vector<int> c) {
	// Add contents from vector 'a' and 'b' into vector 'c' || Transform using a Lambda C++11
	std::transform(a.begin(), a.end(), b.begin(), c.begin(), 
		[](int a, int b) {return a + b; });
}