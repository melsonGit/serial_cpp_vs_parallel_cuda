#include "../inc/ProgramUtilities.h"

#include <iostream>
#include <thread>

void ProgramUtilities::clearInputStream() const
{
	std::cin.clear();
	std::cin.ignore(std::numeric_limits<std::streamsize>::max(), '\n');
}
void ProgramUtilities::clearScreen() const
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
const int ProgramUtilities::getInput() const
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
void ProgramUtilities::fakeLoad() const
{
	for (auto i = 0; i < 3; ++i) {
		std::cout << ". ";
		std::this_thread::sleep_for(std::chrono::seconds(1));
	}
}

// Windows.h incuded below as it clashes with getInput()
#include <Windows.h> 

const bool ProgramUtilities::getKeyPress() const
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