#include "../inc/ProgramHandler.h"
#include "../inc/ArithmeticDetails.h"
#include "../inc/ArithmeticOperation.h"
#include "../inc/VectorAddition.h"
#include "../inc/MatrixMultiplication.h"
#include "../inc/OneDConvolution.h"

#include <cassert>
#include <chrono>
#include <iostream>
#include <string_view>
#include <thread>

void ProgramHandler::launchProgram()
{
	bool inMainMenu{ true };

	do // Enter main menu
	{
		this->launchDirective();
		this->userSetDirective(inMainMenu);

		if (this->getDirective() != ProgramDirective::programExit)
		{
			do
			{
				this->launchDirective(); // Launch Operation / Exit back to main menu or exit program

			} while ((this->getDirective() != ProgramDirective::mainMenu) && (this->getDirective() != ProgramDirective::programExit));
		}
	} while (this->getDirective() != ProgramDirective::programExit);

	assert(this->getDirective() == ProgramDirective::programExit && "We should only be here if our directive is set to programExit!");
	this->launchDirective();
}
void ProgramHandler::clearInputStream() const
{
	std::cin.clear();
	std::cin.ignore(std::numeric_limits<std::streamsize>::max(), '\n');
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
const int ProgramHandler::getOpSampleSelection()
{
	using enum ProgramDirective;
	int validSelection{ false };
	int selectionRange{};

	do
	{
		selectionRange = getInput();

		if (selectionRange >= 1 && selectionRange <= 5)
			validSelection = true;
		else
		{
			switch (static_cast<ProgramDirective>(selectionRange))
			{ // should be using sudoSetDirective() here!
			case inOpMainMenu: { this->mDisplay = mainMenu; selectionRange = static_cast<int>(mainMenu); validSelection = true; break; }
			case inOpProgramExit: { this->mDisplay = programExit;  selectionRange = static_cast<int>(programExit); validSelection = true; break; }
			default: { std::cout << "\nInvalid selection!\n\n"; break; }
			}
		}
	} while (!validSelection);

	return selectionRange;
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
void ProgramHandler::displayMainMenu() const
{
	using namespace ArithmeticDetails;
	using enum ProgramDirective;

	std::cout << "\n\n\n\t\t\tRedirecting to Main Menu";

	clearScreen();

	std::cout << "Please select an arithmetic operation from the options below.\n"
		<< "Enter corresponding number to make selection: \n\nArithmetic Operations\n\n"
		<< '[' << static_cast<int>(vectorAddition) << "]\t" << VectorAdditionDetails::vecAddName << '\n'
		<< '[' << static_cast<int>(matrixMultiplication) << "]\t" << MatrixMultiplicationDetails::matMultiName << '\n'
		<< '[' << static_cast<int>(oneConvolution) << "]\t" << OneDConvolutionDetails::oneDConvName << '\n'
		<< '[' << static_cast<int>(twoConvolution) << "]\t" << TwoDConvolutionDetails::twoDConvName << "\n\n"
		<< "Program Navigation\n\n"
		<< '[' << static_cast<int>(mainMenu) << "]\tReturn to Main Menu\n"
		<< '[' << static_cast<int>(programExit) << "]\tClose Program\n";
}
void ProgramHandler::displayOperationName(const ArithmeticOperation& operation) const
{
	std::cout << "\n\n\n\t\t\tLaunching " << operation.getOpName() << " operation";
	clearScreen();
}
void ProgramHandler::displayOperationSampleSelection(const ArithmeticOperation& operation) const
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
void ProgramHandler::displayOpDetails(const ArithmeticOperation& operation) const
{
	displayOperationName(operation);
	displayOperationSampleSelection(operation);
}
void ProgramHandler::displayProgramExit() const
{
	std::cout << "\n\n\n\t\t\tClosing program";
	clearScreen();
}

#include <Windows.h> // Included here as it clashes with getInput()

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
void ProgramHandler::fakeLoad() const
{
	for (int i = 0; i < 3; ++i) {
		std::cout << ". ";
		std::this_thread::sleep_for(std::chrono::seconds(1));
	}
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
void ProgramHandler::userSetDirective(const bool& isMainMenu)
{
	using enum ProgramDirective;
	bool validSelection{ false };

	if (isMainMenu)
	{
		do
		{
			switch (static_cast<ProgramDirective>(getInput()))
			{
			case vectorAddition: { this->mDisplay = vectorAddition; validSelection = true; break; }
			case matrixMultiplication: { this->mDisplay = matrixMultiplication; validSelection = true; break; }
			case oneConvolution: { this->mDisplay = oneConvolution; validSelection = true; break; }
			case twoConvolution: { this->mDisplay = twoConvolution; validSelection = true; break; }
			case mainMenu: { this->mDisplay = mainMenu; validSelection = true; break; }
			case programExit: { this->mDisplay = programExit; validSelection = true; break; }
			default: { std::cout << "\nInvalid selection!\n\n"; break; }
			}
		} while (!validSelection);
	}
	else
	{
		do
		{
			switch (static_cast<ProgramDirective>(getInput()))
			{
			case inOpMainMenu: { this->mDisplay = mainMenu; validSelection = true; break; }
			case inOpProgramExit: { this->mDisplay = programExit; validSelection = true; break; }
			default: { std::cout << "\nInvalid selection!\n\n"; break; }
			}
		} while (!validSelection);
	}
}
void ProgramHandler::sudoSetDirective(const ProgramDirective& sudoChoice)
{
	using enum ProgramDirective;
	bool validSudoSelection{ true };

	switch (sudoChoice)
	{
	case vectorAddition: { this->mDisplay = vectorAddition; break; }
	case matrixMultiplication: { this->mDisplay = matrixMultiplication; break; }
	case oneConvolution: { this->mDisplay = oneConvolution; break; }
	case twoConvolution: { this->mDisplay = twoConvolution; break; }
	case mainMenu: { this->mDisplay = mainMenu; break; }
	case programExit: { this->mDisplay = programExit; break; }
	default: { validSudoSelection = false; break; }
	}

	assert(validSudoSelection && "Invalid sudoSetDirective() argument!");
}
void ProgramHandler::launchDirective() // this should be encapsulated into 2/3 functions - bending DRY rules a bit here
{
	using enum ProgramDirective;

	switch (this->mDisplay)
	{
	case programStart: { displayProgramStart(); break; }
	case vectorAddition: 
	{ 
		VectorAddition vecAdd{}; 
		bool validSelection{};

		do
		{
			displayOpDetails(vecAdd);
			validSelection = false;
			int userSampleDisplaySelection{ this->getOpSampleSelection() };

			if (userSampleDisplaySelection == static_cast<int>(mainMenu) || userSampleDisplaySelection == static_cast<int>(programExit))
			{
				validSelection = true;
			}
			else
			{
				vecAdd.startOpSeq(userSampleDisplaySelection);
			}
		} while (!validSelection);

		break; 
	}
	case matrixMultiplication: { MatrixMultiplication matMulti{}; displayOpDetails(matMulti); matMulti.startOpSeq(this->getInput()); break; }
	case oneConvolution: {OneDConvolution oneConv{}; displayOpDetails(oneConv); oneConv.startOpSeq(this->getInput()); break; }
	case mainMenu: { displayMainMenu(); break; }
	case programExit: { displayProgramExit(); break; }
	default: { std::cout << "\nInvalid selection!\n\n"; break; }
	}
}
const ProgramDirective& ProgramHandler::getDirective() const
{
	return this->mDisplay;
}