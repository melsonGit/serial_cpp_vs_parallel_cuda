#pragma once
#ifndef PROGRAM_HANDLER
#define PROGRAM_HANDLER

#include "ArithmeticOperation.h"

#include <cassert>
#include <string_view>

enum class ProgramDirective
{
	programStart,
	vectorAddition,
	matrixMultiplication,
	oneConvolution,
	twoConvolution,
	/*<----Add new operations here---->*/
	programExit,
	mainMenu,
	exitProgramFromOperation,
};

// Handles program events

class ProgramHandler
{
private:

	ProgramDirective mDisplay{};

	void clearScreen() const;
	void displayMainMenu() const;
	void displayProgramExit() const;
	void displayProgramStart() const;
	void displayOperationName(const ArithmeticOperation& operation) const;
	void fakeLoad() const;
	const bool getKeyPress() const;

public:

	ProgramHandler(const ProgramDirective& directive = ProgramDirective::programStart)
		: mDisplay{ directive } 
	{
		if (directive == ProgramDirective::programStart)
		{
			// Display starting screen - set directive to main menu - display our main menu
			launchDirective();
			setDirective(6);
			launchDirective();
		}
		else // We only want to initialise our ProgramHandler object with ProgramDirective::programStart
			assert(directive == ProgramDirective::programStart 
				   && "ProgramHandler object initialised with a ProgramDirective value that is not programStart!");
	}

	ProgramDirective& getDirective();
	void launchDirective() const;
	void setDirective(const int& userInput);
};
#endif