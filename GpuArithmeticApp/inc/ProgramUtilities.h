#pragma once
#ifndef PROGRAM_UTILITIES
#define PROGRAM_UTILITIES

class ProgramUtilities
{
private:

	void clearInputStream() const;
	void fakeLoad() const;

public:

	ProgramUtilities() = default;

	void clearScreen() const;
	const int getInput() const;
	const bool getKeyPress() const;
};
#endif