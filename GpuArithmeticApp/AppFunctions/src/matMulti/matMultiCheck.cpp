#include "../../inc/matMulti/matMultiCheck.h"

// Check result on the CPU
void matMultiCheck(std::vector<int> const& a, std::vector<int> const& b, std::vector<int> const& c, int const& conSize) 
{
	std::cout << "\nMatrix Multiplication: Authenticating results.\n\n";

	bool doesMatch { true };

    // For each row
    for (auto i { 0 }; i < conSize; i++) 
	{
		// For each column in that row
        for (auto j { 0 }; j < conSize && doesMatch; j++) 
		{
			// For each row-column combination
            int resultVar { 0 };

            for (auto k { 0 }; k < conSize; k++) 
			{
                // Accumulate the partial results
                resultVar += a[i * conSize + k] * b[k * conSize + j];
            }

			if (resultVar != c[i * conSize + j])
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