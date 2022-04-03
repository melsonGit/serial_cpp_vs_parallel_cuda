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

	// Display Utilities
	void displayProgramStart() const;
	void displayMainMenu() const;
	void displayOperationName(const ArithmeticOperation& operation) const;
	void displaySampleSelection(const ArithmeticOperation& operation) const;
	void displayProgramExit() const;

	// Core Utilities
	const bool getKeyPress() const;
	void fakeLoad() const;
	void clearScreen() const;

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

	void setDirective(const int& userInput);
	void launchDirective() const;
	const ProgramDirective& getDirective() const;
};
#endif