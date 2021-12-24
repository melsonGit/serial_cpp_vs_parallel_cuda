#include "../../inc/matMulti/matMultiCheck.h"

void matMultiCheck(std::vector<std::vector<int>> const& a, std::vector<std::vector<int>> const& b, std::vector<std::vector<int>> const& c)
{
#if 0
	std::cout << "\nMatrix Multiplication: Authenticating results.\n\n";
	bool doesMatch = true;

	for (auto i{0}; i < a.size(); i++)
	{
		if ((a[i][i] * b[i][i]) != c[i][i])
			doesMatch = false;
		else
			continue;

	}

	if (!doesMatch)
		std::cout << "Matrix multiplication unsuccessful: output vector data does not match the expected result.\n"
		          << "Timing results will be discarded.\n\n";
	else
		std::cout << "Matrix multiplication successful: output vector data matches expected results.\n"
		          << "Timing results will be recorded.\n\n";
#endif
}

// Implement a feature that automatically inputs successful data into an excel spreadsheet - via python script or third-party library