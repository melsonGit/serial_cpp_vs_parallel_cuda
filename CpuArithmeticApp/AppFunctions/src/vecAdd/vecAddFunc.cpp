#include "../../inc/vecAdd/vecAddFunc.h"

void vecAddFunc(std::vector<int> const &a, std::vector<int>  const &b, std::vector<int> &c)
{
	// Add contents from vector 'a' and 'b' into vector 'c' || Transform using a Lambda C++11
	transform(a.begin(), a.end(), b.begin(), c.begin(),
		[](int a, int b) {return a + b; });
}