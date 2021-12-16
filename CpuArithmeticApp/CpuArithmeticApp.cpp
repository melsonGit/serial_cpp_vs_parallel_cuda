#include <iostream>
#include "../CpuArithmeticApp/AppFunctions/inc/allHeaders.h"

int main()
{
	int runProg{}, closeProg{ 5 };

	do 
	{
		runProg = 0;

		opChoice(runProg);

		if (runProg == 1)
			vecCore();
		else if (runProg == 2)
			matMultiCore();
		else if (runProg == 3)
			oneConvCore();
		else if (runProg == 4)
			twoConvCore();

	} while (runProg != closeProg);

	std::cout << "\nClosing program.\n";

	return EXIT_SUCCESS;

}