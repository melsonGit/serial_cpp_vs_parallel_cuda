#include "../../inc/matMulti/matMultiCore.h"

using Clock = std::chrono::steady_clock;

void matMultiCore()
{
	// Assign variable conSize with a user selected value
	int conSize { matMultiConSet(conSize) };

	// Assign 2D input vectors (inputVecA & inputVecB) and the 2D output vector (resultVec) a container size of conSize by 2
	std::vector<std::vector<int>> 
	inputVecA(conSize, std::vector<int>(2, 0)), 
	inputVecB(conSize, std::vector<int>(2, 0)), 
	resultVec(conSize, std::vector<int>(2, 0));

	// Populate vectors
	std::cout << "\nMatrix Multiplication: Populating 1 of 2 input vectors.\n";
	matMultiNumGen(inputVecA);
	std::cout << "\nMatrix Multiplication: Populating 2 of 2 input vectors.\n";
	matMultiNumGen(inputVecB);

	// Start clock
	auto opStart { Clock::now() };

	// Begin sequential matrix multiplication operation
	matMultiFunc(inputVecA, inputVecB, resultVec, conSize);

	// Stop clock
	auto opEnd { Clock::now() };

	// Check output vector contents
	matMultiCheck(inputVecA, inputVecB, resultVec, conSize);

	// Output timing to complete operation and container size
	std::cout << "CPU Matrix Multiplication computation time (container size: " << conSize * 2 << "):\n"
			  << std::chrono::duration_cast<std::chrono::microseconds>(opEnd - opStart).count() << " us\n"
			  << std::chrono::duration_cast<std::chrono::milliseconds>(opEnd - opStart).count() << " ms\n\n"
			  << "Returning to selection screen.\n\n"

			  << "#########################################################################\n" <<
				 "#########################################################################\n" <<
				 "#########################################################################\n\n";
}