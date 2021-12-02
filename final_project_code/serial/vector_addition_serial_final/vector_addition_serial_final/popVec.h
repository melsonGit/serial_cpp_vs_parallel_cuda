#pragma once

#include "popVec.cpp"
#include <iostream>
#include <cstdlib>

class popVec
{
private:
	int elementSize{};
public:
	// Constructor
	popVec(int x);

	// Function to assign elementSize a value
	int elementSet(int&);
};
