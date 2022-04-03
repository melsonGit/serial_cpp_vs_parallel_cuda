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

	// Navigates our ProgramDirective functions
	ProgramDirective mDisplay{};

	// Display Utilities
	void startProgram() const;
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
			// 1. Display starting screen 2. Set directive to main menu 3. Display our main menu
			launchDirective();
			setDirective(6);
			launchDirective();
		}
		else // Only initialise ProgramHandler object with ProgramDirective::programStart
			assert(directive == ProgramDirective::programStart 
				   && "ProgramHandler object initialised with a ProgramDirective value that is not programStart!");
	}

	void setDirective(const int& userInput);
	void launchDirective() const;
	const ProgramDirective& getDirective() const;
};
#endif