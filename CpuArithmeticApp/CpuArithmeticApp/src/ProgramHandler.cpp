#include "../inc/ProgramHandler.h"
#include "../inc/ArithmeticOperation.h"
#include "../inc/VectorAddition.h"
#include "../inc/MatrixMultiplication.h"

#include <chrono>
#include <iostream>
#include <string_view>
#include <thread>
#include <Windows.h>

// Ordered alphabetically

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
}

void ProgramHandler::displayProgramExit() const
{
	std::cout << "\n\n\n\t\t\tClosing program";
	clearScreen();
}

void ProgramHandler::displayProgramStart() const
{
	std::cout << "\n\n\n\n\n\t\t\t|/| CPU vs GPU Arithmetic App |\\|\n"
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

void ProgramHandler::displayOperationName(const ArithmeticOperation& operation) const
{
	std::cout << "\n\n\n\t\t\tLaunching " << operation.getOperationName() << " operation";
	clearScreen();
}

void ProgramHandler::fakeLoad() const
{
	for (int i = 0; i < 3; ++i) {
		std::cout << ". ";
		std::this_thread::sleep_for(std::chrono::seconds(1));
	}
}

ProgramDirective& ProgramHandler::getDirective()
{
	return mDisplay;
}

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

void ProgramHandler::launchDirective() const
{
	using enum ProgramDirective;

	switch (mDisplay)
	{
	case programStart: { displayProgramStart(); break; }
	case mainMenu: { displayMainMenu(); break; }
	case programExit: { displayProgramExit(); break; }
	case vectorAddition: { VectorAddition vecAdd{}; displayOperationName(vecAdd); vecAdd.startOperationSequence(); break; }
	case matrixMultiplication: { MatrixMultiplication matMulti{}; displayOperationName(matMulti); matMulti.startOperationSequence(); break; }
	case oneConvolution:
	{
		std::cout << "\n\n\n\t\t\tLaunching 1D Convolution operation";
		clearScreen();
		// start oneConv op
		break;
	}
	case twoConvolution:
	{
		std::cout << "\n\n\n\t\t\tLaunching 2D Convolution operation";
		clearScreen();
		// start twoConv op
		break;
	}
	default: { std::cout << "\nInvalid selection!\n\n"; break; }
	}
}

void ProgramHandler::setDirective(const int& userInput)
{
	using enum ProgramDirective;

	switch (static_cast<ProgramDirective>(userInput))
	{
	case mainMenu: { mDisplay = mainMenu; break; }
	case vectorAddition: { mDisplay = vectorAddition; break; }
	case matrixMultiplication: { mDisplay = matrixMultiplication; break; }
	case oneConvolution: { mDisplay = oneConvolution; break; }
	case twoConvolution: { mDisplay = twoConvolution; break; }
	case programExit: { mDisplay = programExit; break; }
	default: { std::cout << "\nInvalid selection!\n\n"; break; }
	}
}