#include "../../inc/matMulti/matMultiCheck.h"

void matMultiCheck(std::vector<int> const& a, std::vector<int> const& b, std::vector<int> const& c, int conSize) 
{
	std::cout << "\nMatrix Multiplication: Authenticating results.\n\n";

	bool doesMatch { true };

    // For each row
    for (auto rowIn { 0 }; rowIn < conSize; rowIn++) 
	{
		// For each column in that row
        for (auto colIn { 0 }; colIn < conSize && doesMatch; colIn++) 
		{
			// For each row-column combination
            int resultVar { 0 };

            for (auto rowColPair { 0 }; rowColPair < conSize; rowColPair++) 
			{
                // Accumulate results in that combo
                resultVar += a[rowIn * conSize + rowColPair] * b[rowColPair * conSize + colIn];
            }

			if (resultVar != c[rowIn * conSize + colIn])
				doesMatch = false;
			else
				continue;
        }
    }
	if (!doesMatch)
		std::cout << "Matrix multiplication unsuccessful: output vector data does not match expected results.\n"
		<< "Timing results will be discarded.\n";
	else
		std::cout << "Matrix multiplication successful: output vector data matches expected results.\n"
		<< "Timing results will be recorded.\n";
}