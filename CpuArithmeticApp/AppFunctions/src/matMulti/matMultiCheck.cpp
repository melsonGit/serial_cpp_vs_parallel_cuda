#include "../../inc/matMulti/matMultiCheck.h"

void matMultiCheck(std::vector<std::vector<int>> const& a, std::vector<std::vector<int>> const& b, 
				   std::vector<std::vector<int>> const& c, int const& numRows)
{
	std::cout << "\nMatrix Multiplication: Authenticating results.\n\n";

	bool doesMatch { true };

	// Only 2 columns exist in a 2D vector
	int numCols { 2 };

	// For each row
	for (auto i { 0 }; i < numRows; i++)
	{
		// For each column in that row
		for (auto j { 0 }; j < numCols && doesMatch; j++)
		{
			// For each row-column combination
			int resultVar { 0 };

			for (auto k { 0 }; k < numCols; k++)
			{
				resultVar += a[i][k] * b[k][j];
			}

			if (resultVar != c[i][j])
				doesMatch = false;
			else
				continue;
		}
	}

	if (!doesMatch)
		std::cout << "Matrix multiplication unsuccessful: output vector data does not match expected results.\n"
		          << "Timing results will be discarded.\n\n";
	else
		std::cout << "Matrix multiplication successful: output vector data matches expected results.\n"
		          << "Timing results will be recorded.\n\n";
}