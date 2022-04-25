#pragma once
#ifndef CHECK_TWO_CONV
#define CHECK_TWO_CONV

#include "../maskAttributes/maskAttributes.h"

#include <cassert>
#include <iostream>

void twoConvCheck(const int* mainVec, const int* maskVec, const int* resultVec, const int& conSize);

#endif