#include "../../inc/vecAdd/vecAddFunc.h"

void vecAddFunc(std::vector<int> const &inputVecA, std::vector<int> const &inputVecB, std::vector<int> &resultVec)
{
	std::cout << "\nVector Addition: Populating complete.\n";
	std::cout << "\nVector Addition: Starting operation.\n";

	// Add contents from inputVecA and inputVecB into resultVec || Transform using a Lambda C++11
	transform(inputVecA.begin(), inputVecA.end(), inputVecB.begin(), resultVec.begin(),
		[](int a, int b) {return a + b; });

	std::cout << "\nVector Addition: Operation complete.\n";
}