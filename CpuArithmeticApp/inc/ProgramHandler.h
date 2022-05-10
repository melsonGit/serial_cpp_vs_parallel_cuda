#pragma once
#ifndef PROGRAM_HANDLER
#define PROGRAM_HANDLER

#include "ArithmeticOperation.h"
#include "ProgramDirectives.h"

#include <cassert>
#include <string_view>

// Handles program events
class ProgramHandler
{
private:

	// Navigates our ProgramDirectives functions
	ProgramDirectives mDirectiveId{};

	// ProgramHandler Utilites (Windows.h incuded below as it clashes with getInput()) - move into separate class (ProgramUtilites)
	void clearInputStream() const;
	void clearScreen() const;
	const int getInput() const;
	void fakeLoad() const;
	const bool getKeyPress() const;

	// Display Non-Arithmetic Operation EventDirectives
	void displayMainMenu() const;
	void displayProgramExit() const;
	void displayProgramStart() const;

	// Display Arithmetic Operation EventDirectives
	void displayOperationDetails(const ArithmeticOperation& operation) const;
	void displayOperationName(const ArithmeticOperation& operation) const;
	void displayOperationSampleMenu(const ArithmeticOperation& operation) const;

	// User Input EventDirectives
	const int userOpSampleSelection();
	void userSetMainMenuDirective();

	// Directive Getters/Setters
	const ProgramDirectives& getCurrentDirective() const;
	void setSudoDirective(const ProgramDirectives&);

	// Launch Directive
	void processOperationDirecitve(ArithmeticOperation& operation);
	void launchDirective();
	
public:

	ProgramHandler(const ProgramDirectives& directive = ProgramDirectives::programStart)
		: mDirectiveId{ directive } 
	{
		if (directive == ProgramDirectives::programStart)
		{
			// 1. Display starting screen 2. Set directive to main menu
			this->launchDirective();
			this->setSudoDirective(ProgramDirectives::mainMenu);
		}
		else // Only initialise ProgramHandler object with ProgramDirectives::programStart
			assert(directive == ProgramDirectives::programStart 
				   && "ProgramHandler object initialised with a ProgramDirectives value that is not programStart!");
	}

	// Program loop starts and ends exists here
	void launchProgram();
};
#endif