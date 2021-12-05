#include "../inc/checkAdd.h"

void checkAdd(std::vector<int> const& a, std::vector<int> const& b, std::vector<int> const& c)
{
	std::cout << "Checking results..." << std::endl;
	bool addsMatch = true;

	for (int i{0}; i < a.size(); i++)
	{
		if ((a[i] + b[i]) != c[i])
			addsMatch = false;
		else
			continue;

	}
	
	if (!addsMatch)
		std::cout << "Vector addition unsuccessful; output vector data does not match the expected result.\n"
				  << "Timing results will be discarded.\n" << std::endl;
	else
		std::cout << "Vector addition successful; output vector data matches expected results.\n"
				  << "Timing results will be recorded.\n" << std::endl;

}

// Implement a feature that automatically inputs successful data into an excel spreadsheet - via python script or third-party library