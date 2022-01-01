#include "../CpuArithmeticApp/AppFunctions/inc/allHeaders.h"

int main()
{
	int runProg { 0 }, closeProg { 5 };

	do 
	{
		runProg = 0;

		opChoice(runProg);

		switch (runProg)
		{
		case 1:
			vecAddCore();
			break;
		case 2:
			matMultiCore();
			break;
		case 3:
			oneConvCore();
			break;
		case 4:
			twoConvCore();
			break;
		default:
			break;
		}
	} while (runProg != closeProg);

	return EXIT_SUCCESS;

}