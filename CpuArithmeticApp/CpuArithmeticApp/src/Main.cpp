#include "../inc/ProgramHandler.h"
#include "../inc/ArithmeticOperation.h"

// select op
// generate the container
// execute op
// review op
// close or another op

#include <iostream>

// static global as we only need one - this begins our program
static ProgramHandler handler{};

int main()
{
	do // Enter main menu
	{
		int choice{};
		std::cout << "\n\nOperation Selection: Please select an operation to execute:";
		std::cin >> choice;

		handler.setDirective(choice);

		do // Enter sample size menu here
		{
			std::cout << "\n\nSample Selection: Please select an operation to execute:";

			handler.launchDirective();
			
		} while (handler.getDirective() != ProgramDirective::mainMenu); // maybe add check to see when operation is complete? this is a design decision

	} while (handler.getDirective() != ProgramDirective::programExit);

	EXIT_SUCCESS;
}