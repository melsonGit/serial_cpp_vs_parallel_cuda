#include "../../inc/vecAdd/vecNumGen.h"

void vecNumGen(std::vector<int> &a, std::vector<int> &b)
{
	std::cout << "\nVector Addition: Populating input vectors.\n";

	// Re-seed rand() function for each run
	srand((uint64_t)time(NULL));

	// Generate random numbers via Lambda C++11 function, and place into vector
	std::generate(a.begin(), a.end(), []() { return rand() % 100; });
	std::generate(b.begin(), b.end(), []() { return rand() % 100; });

	std::cout << "\nVector Addition: Populating complete.\n";
}