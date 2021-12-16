#include "../../inc/vecAdd/vecCheck.h"

void vecCheck(std::vector<int> const& a, std::vector<int> const& b, std::vector<int> const& c)
{
	std::cout << "Checking results..." << std::endl;
	bool doesMatch = true;

	for (int i{0}; i < a.size(); i++)
	{
		if ((a[i] + b[i]) != c[i])
			doesMatch = false;
		else
			continue;

	}
	
	if (!doesMatch)
		std::cout << "Vector addition unsuccessful; output vector data does not match the expected result.\n"
				  << "Timing results will be discarded.\n" << std::endl;
	else
		std::cout << "Vector addition successful; output vector data matches expected results.\n"
				  << "Timing results will be recorded.\n" << std::endl;

}

// Implement a feature that automatically inputs successful data into an excel spreadsheet - via python script or third-party library