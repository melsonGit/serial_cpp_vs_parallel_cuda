#pragma once
#ifndef PROGRAM_DIRECTIVES
#define PROGRAM_DIRECTIVES

#include <unordered_map>
#include <string>

// Ignore Error C26812: Enum Class values aren't implicitly cast to int, which is what we want
enum ProgramDirectives
{
	programStart = 0,
	vectorAddition,
	matrixMultiplication,
	oneConvolution,
	twoConvolution, // <-- New operations below here
	programExit,
	mainMenu,

	// Navigation options within operation sample size selection only - never setSudo/setUser directives to these
	inOpMainMenu = 6,
	inOpProgramExit,
};
#endif