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
	using enum ProgramDirective;

	do // Enter main menu
	{
		this->launchDirective();
		this->userSetMainMenuDirective();

		if (this->getDirective() != programExit)
		{
			do
			{
				this->launchDirective(); // Launch Operation / Exit back to main menu or exit program

			} while ((this->getDirective() != mainMenu) && (this->getDirective() != programExit));
		}
	} while (this->getDirective() != programExit);

	assert(this->getDirective() == programExit && "We should only be here if our directive is set to programExit!");
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
	fakeLoad();

	// String of special characters (translates to clear screen command) that clears CMD window
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
			std::cout << "Please enter numbers only.\n";
			clearInputStream();
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
	for (int i = 0; i < 3; ++i) {
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

// Display Non-Arithmetic Operation Events
void ProgramHandler::displayMainMenu() const
{
	using namespace ArithmeticDetails;
	using enum ProgramDirective;

	std::cout << "\n\n\t\t\tRedirecting to Main Menu";

	clearScreen();

	std::cout << "Please select an arithmetic operation from the options below.\n"
		<< "Enter corresponding number to make selection: \n\nArithmetic Operations\n\n"
		<< '[' << static_cast<int>(vectorAddition) << "]\t" << VectorAdditionDetails::vecAddName << '\n'
		<< '[' << static_cast<int>(matrixMultiplication) << "]\t" << MatrixMultiplicationDetails::matMultiName << '\n'
		<< '[' << static_cast<int>(oneConvolution) << "]\t" << OneDConvolutionDetails::oneDConvName << '\n'
		<< '[' << static_cast<int>(twoConvolution) << "]\t" << TwoDConvolutionDetails::twoDConvName << "\n\n"
		<< '[' << static_cast<int>(programExit) << "]\tClose Program\n";
}
void ProgramHandler::displayProgramExit() const
{
	std::cout << "\n\n\t\t\tShutting down program";
	clearScreen();
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
	while (!getKeyPress());
}

// Display Arithmetic Operation Events
void ProgramHandler::displayOperationDetails(const ArithmeticOperation& operation) const
{
	displayOperationName(operation);
	displayOperationSampleMenu(operation);
}
void ProgramHandler::displayOperationName(const ArithmeticOperation& operation) const
{
	std::cout << "\n\n\t\t\tLaunching " << operation.getOpName() << " operation";
	clearScreen();
}
void ProgramHandler::displayOperationSampleMenu(const ArithmeticOperation& operation) const
{
	using enum ProgramDirective;

	int elementOptions{ 0 };

	std::cout << "Choose " << operation.getOpName() << " element sample size from the options below.\n"
		<< "Enter corresponding number to make selection: \n\nSample Sizes\n\n"
		<< "[1] " << operation.getOpSampleSize(elementOptions++) << " elements\n"
		<< "[2] " << operation.getOpSampleSize(elementOptions++) << " elements\n"
		<< "[3] " << operation.getOpSampleSize(elementOptions++) << " elements\n"
		<< "[4] " << operation.getOpSampleSize(elementOptions++) << " elements\n"
		<< "[5] " << operation.getOpSampleSize(elementOptions) << " elements\n\n"
		<< "Program Navigation\n\n"
		<< '[' << static_cast<int>(inOpMainMenu) << "]\tReturn to Main Menu\n"
		<< '[' << static_cast<int>(inOpProgramExit) << "]\tClose Program\n";
}

// User Input Events
const int ProgramHandler::userOpSampleSelection()
{
	using enum ProgramDirective;
	bool validSelection{ false };
	int selectionRange{};

	do
	{
		selectionRange = getInput();
		// If user selections within sample range 1 - 5, we return that value
		if (selectionRange >= 1 && selectionRange <= 5)
			validSelection = true;
		// If the user selection is outside sample range, we check against our directives
		else
		{
			switch (static_cast<ProgramDirective>(selectionRange))
			{
			case inOpMainMenu: { this->setSudoDirective(mainMenu); selectionRange = static_cast<int>(inOpMainMenu); validSelection = true; break; }
			case inOpProgramExit: { this->setSudoDirective(programExit);  selectionRange = static_cast<int>(inOpProgramExit); validSelection = true; break; }
			default: { std::cout << "\nInvalid selection!\n\n"; break; }
			}
		}
	} while (!validSelection);

	return selectionRange;
}
void ProgramHandler::userSetMainMenuDirective()
{
	using enum ProgramDirective;
	bool validSelection{ false };

	do
	{
		switch (static_cast<ProgramDirective>(getInput()))
		{
		case vectorAddition: { this->mDisplay = vectorAddition; validSelection = true; break; }
		case matrixMultiplication: { this->mDisplay = matrixMultiplication; validSelection = true; break; }
		case oneConvolution: { this->mDisplay = oneConvolution; validSelection = true; break; }
		case twoConvolution: { this->mDisplay = twoConvolution; validSelection = true; break; }
		case programExit: { this->mDisplay = programExit; validSelection = true; break; }
		default: { std::cout << "\nInvalid selection!\n\n"; break; }
		}
	} while (!validSelection);
}

// Directive Getters/Setters
const ProgramDirective& ProgramHandler::getDirective() const
{
	return this->mDisplay;
}
void ProgramHandler::setSudoDirective(const ProgramDirective& sudoChoice)
{
	using enum ProgramDirective;
	bool validSudoSelection{ true };

	switch (sudoChoice)
	{
	case vectorAddition: { this->mDisplay = vectorAddition; break; }
	case matrixMultiplication: { this->mDisplay = matrixMultiplication; break; }
	case oneConvolution: { this->mDisplay = oneConvolution; break; }
	case twoConvolution: { this->mDisplay = twoConvolution; break; }
	case programExit: { this->mDisplay = programExit; break; }
	case mainMenu: { this->mDisplay = mainMenu; break; }
	default: { validSudoSelection = false; break; }
	}

	assert(validSudoSelection && "Invalid setSudoDirective() argument!");
}

// Launch Directive
void ProgramHandler::launchDirective() // this should be encapsulated into 2/3 functions - bending DRY rules a bit here
{
	using enum ProgramDirective;

	switch (this->mDisplay)
	{
	case programStart: { displayProgramStart(); break; }
	case vectorAddition:
	{
		VectorAddition vecAdd{};
		// enterSampleSelectionLoop(const ArithmeticOperation& operation); best encapsulate this
		bool toExitOp{};

		do
		{
			displayOperationDetails(vecAdd);
			toExitOp = false;
			int userSampleDisplaySelection{ this->userOpSampleSelection() };

			// If userSampleDisplaySelection is outside sample selection, user either wants to return to main menu / close program...
			// ...so we skip startOpSeq() and launchDirective()
			if (userSampleDisplaySelection == static_cast<int>(inOpMainMenu) || userSampleDisplaySelection == static_cast<int>(inOpProgramExit))
			{
				toExitOp = true;
			}
			else
			{
				vecAdd.startOpSeq(userSampleDisplaySelection);
			}
		} while (!toExitOp);

		break;
	}
	case matrixMultiplication: 
	{
		MatrixMultiplication matMulti{};
		bool toExitOp{};

		do
		{
			displayOperationDetails(matMulti);
			toExitOp = false;
			int userSampleDisplaySelection{ this->userOpSampleSelection() };

			// If userSampleDisplaySelection is outside sample selection, user either wants to return to main menu / close program...
			// ...so we skip startOpSeq() and launchDirective()
			if (userSampleDisplaySelection == static_cast<int>(inOpMainMenu) || userSampleDisplaySelection == static_cast<int>(inOpProgramExit))
			{
				toExitOp = true;
			}
			else
			{
				matMulti.startOpSeq(userSampleDisplaySelection);
			}
		} while (!toExitOp);

		break;
	}
	case oneConvolution: 
	{
		OneDConvolution oneConv{};
		bool toExitOp{};

		do
		{
			displayOperationDetails(oneConv);
			toExitOp = false;
			int userSampleDisplaySelection{ this->userOpSampleSelection() };

			// If userSampleDisplaySelection is outside sample selection, user either wants to return to main menu / close program...
			// ...so we skip startOpSeq() and launchDirective()
			if (userSampleDisplaySelection == static_cast<int>(inOpMainMenu) || userSampleDisplaySelection == static_cast<int>(inOpProgramExit))
			{
				toExitOp = true;
			}
			else
			{
				oneConv.startOpSeq(userSampleDisplaySelection);
			}
		} while (!toExitOp);

		break;
	}
	case twoConvolution:
	{
		TwoDConvolution twoConv{};
		bool toExitOp{};

		do
		{
			displayOperationDetails(twoConv);
			toExitOp = false;
			int userSampleDisplaySelection{ this->userOpSampleSelection() };

			// If userSampleDisplaySelection is outside sample selection, user either wants to return to main menu / close program...
			// ...so we skip startOpSeq() and launchDirective()
			if (userSampleDisplaySelection == static_cast<int>(inOpMainMenu) || userSampleDisplaySelection == static_cast<int>(inOpProgramExit))
			{
				toExitOp = true;
			}
			else
			{
				twoConv.startOpSeq(userSampleDisplaySelection);
			}
		} while (!toExitOp);

		break;
	}
	case mainMenu: { displayMainMenu(); break; }
	case programExit: { displayProgramExit(); break; }
	default: { std::cout << "\nInvalid selection!\n\n"; break; }
	}
}