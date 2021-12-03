#include "addFunc.h"

void add(std::vector<int> a, std::vector<int> b, std::vector<int> c) {
	// Add contents from vector 'a' and 'b' into vector 'c' || Transform using a Lambda C++11
	transform(a.begin(), a.end(), b.begin(), c.begin(),
		[](int a, int b) {return a + b; });
}