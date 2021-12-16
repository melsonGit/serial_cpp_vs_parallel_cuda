#include "../../inc/vecAdd/vecCheck.h"

void vecCheck(std::vector<int> const& a, std::vector<int> const& b, std::vector<int> const& c)
{
	std::cout << "\nVector Addition: Authenticating results.\n\n";
	bool doesMatch = true;

	for (auto i{0}; i < a.size(); i++)
	{
		if ((a[i] + b[i]) != c[i])
			doesMatch = false;
		else
			continue;

	}
	
	if (!doesMatch)
		std::cout << "Vector addition unsuccessful: output vector data does not match the expected result.\n"
		          << "Timing results will be discarded.\n\n";
	else
		std::cout << "Vector addition successful: output vector data matches expected results.\n"
		          << "Timing results will be recorded.\n\n";
}

// Implement a feature that automatically inputs successful data into an excel spreadsheet - via python script or third-party library