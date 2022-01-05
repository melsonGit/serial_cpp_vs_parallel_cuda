#include "../../inc/matMulti/matMultiCore.h"

void matMultiCore()
{
	// Assign variable conSize with a user selected value
	int conSize { matMultiConSet(conSize) };

	// Assign 2D input vectors (a & b) and the 2D output vector (c) a container size of conSize by 2
	std::vector<std::vector<int>> a(conSize, std::vector<int>(2, 0)), b(conSize, std::vector<int>(2, 0)), c(conSize, std::vector<int>(2, 0));

	// Populate vectors
	std::cout << "\nMatrix Multiplication: Populating 1 of 2 input vectors.\n";
	matMultiNumGen(a);
	std::cout << "\nMatrix Multiplication: Populating 2 of 2 input vectors.\n";
	matMultiNumGen(b);

	// Start clock
	clock_t opStart { clock() };

	// Begin sequential matrix multiplication operation
	matMultiFunc(a, b, c, conSize);

	// Stop clock
	clock_t opEnd { clock() };

	// Check output vector contents
	matMultiCheck(a, b, c, conSize);

	// Calculate overall time spent to complete operation
	double completionTime { (opEnd - opStart) / (double)CLOCKS_PER_SEC };

	// Output timing to complete operation and container size
	std::cout << completionTime << "s Matrix Multiplication computation time, with a container size of " << conSize * 2 << ".\n\n";
	std::cout << "Returning to selection screen.\n\n";

	std::cout << "#########################################################################\n" <<
				 "#########################################################################\n" <<
				 "#########################################################################\n\n";
}