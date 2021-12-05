#include "addFunc.h"
#include <iostream>

void add(std::vector<int> const &a, std::vector<int>  const &b, std::vector<int> &c) {
	// Add contents from vector 'a' and 'b' into vector 'c' || Transform using a Lambda C++11
	transform(a.begin(), a.end(), b.begin(), c.begin(),
		[](int a, int b) {return a + b; });

	// Test - remove later
	// std::cout << a[5001] << '\n' << b[5001] << '\n' << c[5001] << std::endl << std::flush;
}