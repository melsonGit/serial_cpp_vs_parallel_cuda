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
					// <-- Next new operation here
	programExit,
	mainMenu,

	// Navigation options within operation sample size selection only
	inOpMainMenu = 6,
	inOpProgramExit,
};

// Handles program events
class ProgramHandler
{
private:

	// Navigates our ProgramDirective functions
	ProgramDirective mDisplay{};

	// ProgramHandler Utilites (Windows.h incuded below as it clashes with getInput())
	void clearInputStream() const;
	void clearScreen() const;
	const int getInput() const;
	void fakeLoad() const;
	const bool getKeyPress() const;

	// Display Non-Arithmetic Operation Events
	void displayMainMenu() const;
	void displayProgramExit() const;
	void displayProgramStart() const;

	// Display Arithmetic Operation Events
	void displayOperationDetails(const ArithmeticOperation& operation) const;
	void displayOperationName(const ArithmeticOperation& operation) const;
	void displayOperationSampleMenu(const ArithmeticOperation& operation) const;

	// User Input Events
	const int userOpSampleSelection();
	void userSetMainMenuDirective();

	// Directive Getters/Setters
	const ProgramDirective& getDirective() const;
	void setSudoDirective(const ProgramDirective&);

	// Launch Directive
	void launchDirective();
	
public:

	ProgramHandler(const ProgramDirective& directive = ProgramDirective::programStart)
		: mDisplay{ directive } 
	{
		if (directive == ProgramDirective::programStart)
		{
			// 1. Display starting screen 2. Set directive to main menu
			launchDirective();
			setSudoDirective(ProgramDirective::mainMenu);
		}
		else // Only initialise ProgramHandler object with ProgramDirective::programStart
			assert(directive == ProgramDirective::programStart 
				   && "ProgramHandler object initialised with a ProgramDirective value that is not programStart!");
	}

	// Program loop starts and ends exists here
	void launchProgram();
};
#endif