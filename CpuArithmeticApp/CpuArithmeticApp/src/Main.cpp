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
	handler.launchProgram();
	EXIT_SUCCESS;
}