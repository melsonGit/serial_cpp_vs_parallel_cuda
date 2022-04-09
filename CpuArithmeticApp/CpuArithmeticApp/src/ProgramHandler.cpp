#include "../inc/ProgramHandler.h"
#include "../inc/ArithmeticOperation.h"
#include "../inc/VectorAddition.h"
#include "../inc/MatrixMultiplication.h"

#include <cassert>
#include <chrono>
#include <iostream>
#include <string_view>
#include <thread>

void ProgramHandler::launchProgram()
{
	do // Enter main menu
	{
		this->launchDirective();
		this->userSetDirective();

		do // Enter sample size menu here
		{
			this->launchDirective();
			this->userSetDirective();
			this->launchDirective();

		} while (this->getDirective() != ProgramDirective::mainMenu);

	} while (this->getDirective() != ProgramDirective::programExit);
}

void ProgramHandler::clearInputStream() const
{
	std::cin.clear();
	std::cin.ignore(std::numeric_limits<std::streamsize>::max(), '\n');
}

const int& ProgramHandler::getInput() const
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
	std::cout << "\n\n\n\t\t\tRedirecting to Main Menu";

	clearScreen();

	std::cout << "Please select an arithmetic operation from the options below:\n\n"
		<< "Vector Addition:           enter '1'\n"
		<< "Matrix Multiplication:     enter '2'\n"
		<< "1D Convolution:            enter '3'\n"
		<< "2D Convolution:            enter '4'\n\n"
		<< "If you wish to close this program, please enter '5'\n";

	// getter for retrieving op names?
}

void ProgramHandler::displayOperationName(const ArithmeticOperation& operation) const
{
	std::cout << "\n\n\n\t\t\tLaunching " << operation.getOperationName() << " operation";
	clearScreen();
}

void ProgramHandler::displayOperationSampleSelection(const ArithmeticOperation& operation) const
{
	int elementOptions{ 1 };

	std::cout << "Choose " << operation.getOperationName() << " element sample size from the options below.\n"
		<< "Enter corresponding number to make selection: \n\n"
		<< "[1] " << operation.getOperationSampleSize(elementOptions++) << " elements\n"
		<< "[2] " << operation.getOperationSampleSize(elementOptions++) << " elements\n"
		<< "[3] " << operation.getOperationSampleSize(elementOptions++) << " elements\n"
		<< "[4] " << operation.getOperationSampleSize(elementOptions++) << " elements\n"
		<< "[5] " << operation.getOperationSampleSize(elementOptions) << " elements\n";
}

void ProgramHandler::displayOperationDetails(const ArithmeticOperation& operation) const
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

void ProgramHandler::userSetDirective()
{
	using enum ProgramDirective;
	bool validSelection{ false };

	do
	{
		switch (static_cast<ProgramDirective>(getInput()))
		{
		case mainMenu: { mDisplay = mainMenu; validSelection = true; break; }
		case vectorAddition: { mDisplay = vectorAddition; validSelection = true; break; }
		case matrixMultiplication: { mDisplay = matrixMultiplication; validSelection = true; break; }
		case oneConvolution: { mDisplay = oneConvolution; validSelection = true; break; }
		case twoConvolution: { mDisplay = twoConvolution; validSelection = true; break; }
		case programExit: { mDisplay = programExit; validSelection = true; break; }
		default: { std::cout << "\nInvalid selection!\n\n"; break; }
		}

	} while (!validSelection);
}

void ProgramHandler::sudoSetDirective(const ProgramDirective& sudoChoice)
{
	using enum ProgramDirective;
	bool validSudoSelection{ true };

	switch (sudoChoice)
	{
	case mainMenu: { mDisplay = mainMenu; break; }
	case vectorAddition: { mDisplay = vectorAddition; break; }
	case matrixMultiplication: { mDisplay = matrixMultiplication; break; }
	case oneConvolution: { mDisplay = oneConvolution; break; }
	case twoConvolution: { mDisplay = twoConvolution; break; }
	case programExit: { mDisplay = programExit; break; }
	default: { validSudoSelection = false; break; }
	}

	assert(validSudoSelection && "Invalid sudoSetDirective() argument!");
}

void ProgramHandler::launchDirective() const
{
	using enum ProgramDirective;

	switch (mDisplay)
	{
	case programStart: { displayProgramStart(); break; }
	case mainMenu: { displayMainMenu(); break; }
	case programExit: { displayProgramExit(); break; }
	case vectorAddition: { VectorAddition vecAdd{}; displayOperationDetails(vecAdd); vecAdd.startOperationSequence(); break; }
	case matrixMultiplication: { MatrixMultiplication matMulti{}; displayOperationDetails(matMulti); matMulti.startOperationSequence(); break; }
	default: { std::cout << "\nInvalid selection!\n\n"; break; }
	}
}

const ProgramDirective& ProgramHandler::getDirective() const
{
	return mDisplay;
}