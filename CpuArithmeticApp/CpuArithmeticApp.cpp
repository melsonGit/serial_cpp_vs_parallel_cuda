#include <iostream>
#include <algorithm>
#include <vector>
#include "../CpuArithmeticApp/AppFunctions/inc/allHeaders.h"
#include "../CpuArithmeticApp/AppFunctions/inc/allTDefs.h"

int main()
{
	int runProg{}, closeProg{ 5 };

	do 
	{
		runProg = 0;

		opChoice(runProg);

	#if 0
		if (runProg == 1)
			// launchVecAdd();
			int vec;
		else if (runProg == 2)
			// launchMatMulti();
			int multi;
		else if (runProg == 3)
			// launch1d();
			int oneD;
		else if (runProg == 4)
			// launch2d();
			int twoD;
	#endif 

	} while (runProg != closeProg);

	std::cout << "\nClosing program.\n";
	//return EXIT_SUCCESS;

	// Assign variable conSize with a user selected value
	vecAddConSize conSize = conSet(conSize);

	// Assign input vectors (a & b) and the output vector (c) a container size of conSize
	std::vector<int> a(conSize), b(conSize), c(conSize);

	// Populate vectors
	numGen(a, b);

	// Start clock
	clock_t opStart = clock();

	// Begin sequential vector addition operation
	add(a, b, c);

	// Stop clock
	clock_t opEnd = clock();

	// Check output vector contents
	checkAdd(a, b, c);

	// Calculate overall time spent to complete operation
	double completionTime = (opEnd - opStart) / (double)CLOCKS_PER_SEC;

	// Output timing to complete operation and container size
	std::cout << completionTime << "s Vector Addition computation time, with a container size of " << conSize << ".\n";
	std::cout << "Closing program...\n";

	return EXIT_SUCCESS;

}
