#include "../inc/ProgramHandler.h"
#include "../inc/ArithmeticOperation.h"

// select op
// generate the container
// execute op
// review op
// close or another op

// static global as we only need one - this begins our program
static ProgramHandler handler{};

int main()
{
	int choice{ 1 };

	do // Enter main menu
	{
		handler.setDirective(choice);

		do // Enter sample size menu here
		{
			handler.launchDirective();
			
		} while (handler.getDirective() != ProgramDirective::mainMenu); // maybe add check to see when operation is complete? this is a design decision

	} while (handler.getDirective() != ProgramDirective::programExit);

	EXIT_SUCCESS;
}