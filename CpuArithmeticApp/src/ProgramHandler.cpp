#include "../inc/ProgramHandler.h"
#include "../inc/ArithmeticDetails.h"
#include "../inc/ArithmeticOperation.h"
#include "../inc/VectorAddition.h"
#include "../inc/MatrixMultiplication.h"
#include "../inc/OneDConvolution.h"
#include "../inc/TwoDConvolution.h"

#include <cassert>
#include <chrono>
#include <iostream>
#include <string_view>
#include <thread>

// Program loop starts and ends exists here
void ProgramHandler::launchProgram()
{
	do // Enter main menu
	{
		this->launchDirective();
		this->userSetMainMenuDirective();

		if (this->getCurrentDirective() != ProgramDirectives::programExit)
		{
			do
			{
				this->launchDirective(); // Launch Operation / Exit back to main menu or exit program

			} while ((this->getCurrentDirective() != ProgramDirectives::mainMenu) && (this->getCurrentDirective() != ProgramDirectives::programExit));
		}
	} while (this->getCurrentDirective() != ProgramDirectives::programExit);

	assert(this->getCurrentDirective() == ProgramDirectives::programExit && "We should only be here if our directive is set to programExit!");
	this->launchDirective();
}

// ProgramHandler Utilites (Windows.h incuded below as it clashes with getInput())
void ProgramHandler::clearInputStream() const
{
	std::cin.clear();
	std::cin.ignore(std::numeric_limits<std::streamsize>::max(), '\n');
}
void ProgramHandler::clearScreen() const
{
	this->fakeLoad();

	// Special VT100 escape codes to clear a CMD screen
#ifdef _WIN32
	std::cout << "\033[2J\033[1;1H";
#elif defined __linux__
	std::cout << "\033[2J\033[1;1H";
#elif defined __APPLE__
	// do something for mac
#endif
}
const int ProgramHandler::getInput() const
{
	int userInput{ 0 };
	bool validSelection{ false };

	do
	{
		if (!(std::cin >> userInput))
		{
			std::cout << "Please enter numbers only.\n\n";
			this->clearInputStream();
		}
		else
		{
			validSelection = true;
		}
	} while (!validSelection);

	return userInput;
}
void ProgramHandler::fakeLoad() const
{
	for (auto i = 0; i < 3; ++i) {
		std::cout << ". ";
		std::this_thread::sleep_for(std::chrono::seconds(1));
	}
}

#include <Windows.h> 

const bool ProgramHandler::getKeyPress() const
{
#ifdef _WIN32
	bool isKeyPressed{ false };
	while (!isKeyPressed)
	{
		if (GetKeyState(VK_RETURN) & 0x8000) // Prevent progression until user has pressed ENTER/RETURN key
			isKeyPressed = true;
	}

	return isKeyPressed;
#elif defined __linux__
	// do something for linux
#elif defined __APPLE__
	// do something for mac
#endif
}

// Display Non-Arithmetic Operation EventDirectives
void ProgramHandler::displayMainMenu() const
{
	using namespace ArithmeticDetails;

	std::cout << "\n\n\t\t\tRedirecting to Main Menu";

	this->clearScreen();

	std::cout << "Please select an arithmetic operation from the options below.\n"
		<< "Enter corresponding number to make selection: \n\nArithmetic Operations\n\n"
		<< '[' << ProgramDirectives::vectorAddition << "]\t" << VectorAdditionDetails::vecAddName << '\n'
		<< '[' << ProgramDirectives::matrixMultiplication << "]\t" << MatrixMultiplicationDetails::matMultiName << '\n'
		<< '[' << ProgramDirectives::oneConvolution << "]\t" << OneDConvolutionDetails::oneDConvName << '\n'
		<< '[' << ProgramDirectives::twoConvolution << "]\t" << TwoDConvolutionDetails::twoDConvName << "\n\n"
		<< '[' << ProgramDirectives::programExit << "]\tClose Program\n";
}
void ProgramHandler::displayProgramExit() const
{
	std::cout << "\n\n\t\t\tShutting down program";
	this->clearScreen();
}
void ProgramHandler::displayProgramStart() const
{
	std::cout << "\n\n\n\n\t\t\t|/| CPU vs GPU Arithmetic App |\\|\n"
		<< "\t\t\t|/|         Version: CPU      |\\|\n"
		<< "\t\t\t=================================\n"
		<< "\t\t\t Developed for my MSc ICT thesis\n"
		<< "\t\t      This app is in continuous development\n\n"
		<< "\t\t      Follow link below for future updates:\n"
		<< "\t\t\t       github.com/melsonGit\n\n"
		<< "\t\t\tPress Enter/Return key to Proceed";

	// Proceed only when ENTER/RETURN key is pressed
	while (!this->getKeyPress());
}

// Display Arithmetic Operation EventDirectives
void ProgramHandler::displayOperationDetails(const ArithmeticOperation& operation) const
{
	this->displayOperationName(operation);
	this->displayOperationSampleMenu(operation);
}
void ProgramHandler::displayOperationName(const ArithmeticOperation& operation) const
{
	std::cout << "\n\n\t\t\tLaunching " << operation.getOpName() << " operation";
	this->clearScreen();
}
void ProgramHandler::displayOperationSampleMenu(const ArithmeticOperation& operation) const
{
	int elementOptions{ 0 };

	std::cout << "Choose " << operation.getOpName() << " element sample size from the options below.\n"
		<< "Enter corresponding number to make selection: \n\nSample Sizes\n\n"
		<< "[1] " << operation.getOpSampleSize(elementOptions++) << " elements\n"
		<< "[2] " << operation.getOpSampleSize(elementOptions++) << " elements\n"
		<< "[3] " << operation.getOpSampleSize(elementOptions++) << " elements\n"
		<< "[4] " << operation.getOpSampleSize(elementOptions++) << " elements\n"
		<< "[5] " << operation.getOpSampleSize(elementOptions) << " elements\n\n"
		<< "Program Navigation\n\n"
		<< '[' << ProgramDirectives::inOpMainMenu << "]\tReturn to Main Menu\n"
		<< '[' << ProgramDirectives::inOpProgramExit << "]\tClose Program\n";
}

// User Input EventDirectives
const int ProgramHandler::userOpSampleSelection()
{
	bool validSelection{ false };
	int selectionRange{};

	do
	{
		selectionRange = this->getInput();
		// If user selections within sample range 1 - 5, we return that value
		if (selectionRange >= 1 && selectionRange <= 5)
			validSelection = true;
		// If the user selection is outside sample range, we check against our directives
		else
		{
			switch (selectionRange)
			{
			case ProgramDirectives::inOpMainMenu: { this->setSudoDirective(mainMenu); selectionRange = ProgramDirectives::inOpMainMenu; validSelection = true; break; }
			case ProgramDirectives::inOpProgramExit: { this->setSudoDirective(programExit);  selectionRange = ProgramDirectives::inOpProgramExit; validSelection = true; break; }
			default: { std::cout << "\nInvalid selection!\n\n"; break; }
			}
		}
	} while (!validSelection);

	return selectionRange;
}
void ProgramHandler::userSetMainMenuDirective()
{
	bool validSelection{ false };

	do
	{
		switch (this->getInput())
		{
		case ProgramDirectives::vectorAddition: { this->mDirectiveId = ProgramDirectives::vectorAddition; validSelection = true; break; }
		case ProgramDirectives::matrixMultiplication: { this->mDirectiveId = ProgramDirectives::matrixMultiplication; validSelection = true; break; }
		case ProgramDirectives::oneConvolution: { this->mDirectiveId = ProgramDirectives::oneConvolution; validSelection = true; break; }
		case ProgramDirectives::twoConvolution: { this->mDirectiveId = ProgramDirectives::twoConvolution; validSelection = true; break; }
		case ProgramDirectives::programExit: { this->mDirectiveId = ProgramDirectives::programExit; validSelection = true; break; }
		default: { std::cout << "\nInvalid selection!\n\n"; break; }
		}
	} while (!validSelection);
}

// Directive Getters/Setters
const ProgramDirectives& ProgramHandler::getCurrentDirective() const
{
	return this->mDirectiveId;
}
void ProgramHandler::setSudoDirective(const ProgramDirectives& sudoChoice)
{
	bool validSudoSelection{ true };

	switch (sudoChoice)
	{
	case ProgramDirectives::vectorAddition: { this->mDirectiveId = ProgramDirectives::vectorAddition; break; }
	case ProgramDirectives::matrixMultiplication: { this->mDirectiveId = ProgramDirectives::matrixMultiplication; break; }
	case ProgramDirectives::oneConvolution: { this->mDirectiveId = ProgramDirectives::oneConvolution; break; }
	case ProgramDirectives::twoConvolution: { this->mDirectiveId = ProgramDirectives::twoConvolution; break; }
	case ProgramDirectives::programExit: { this->mDirectiveId = ProgramDirectives::programExit; break; }
	case ProgramDirectives::mainMenu: { this->mDirectiveId = ProgramDirectives::mainMenu; break; }
	default: { validSudoSelection = false; break; }
	}

	assert(validSudoSelection && "Invalid setSudoDirective() argument!");
}

// Process Operation Directive
void ProgramHandler::processOperationDirecitve(ArithmeticOperation& operation)
{
	bool toExitOp{};

	do
	{
		this->displayOperationDetails(operation);
		toExitOp = false;
		int userSampleDisplaySelection{ this->userOpSampleSelection() };

		// If userSampleDisplaySelection is outside sample selection, user either wants to return to main menu / close program...
		// ...so we skip startOpSeq() and launchDirective()
		if (userSampleDisplaySelection == ProgramDirectives::inOpMainMenu || userSampleDisplaySelection == ProgramDirectives::inOpProgramExit)
		{
			toExitOp = true;
		}
		else
		{
			operation.startOpSeq(userSampleDisplaySelection);
		}
	} while (!toExitOp);
}

// Launch Directives
void ProgramHandler::launchDirective()
{
	using enum ProgramDirectives;

	switch (this->mDirectiveId)
	{
	case ProgramDirectives::programStart: { this->displayProgramStart(); break; }
	case ProgramDirectives::vectorAddition: { VectorAddition vecAdd{}; this->processOperationDirecitve(vecAdd); break; }
	case ProgramDirectives::matrixMultiplication: { MatrixMultiplication matMulti{}; this->processOperationDirecitve(matMulti); break; }
	case ProgramDirectives::oneConvolution: { OneDConvolution oneConv{}; this->processOperationDirecitve(oneConv); break; }
	case ProgramDirectives::twoConvolution: { TwoDConvolution twoConv{}; this->processOperationDirecitve(twoConv); break; }
	case ProgramDirectives::mainMenu: { displayMainMenu(); break; }
	case ProgramDirectives::programExit: { displayProgramExit(); break; }
	default: { std::cout << "\nInvalid selection!\n\n"; break; }
	}
}