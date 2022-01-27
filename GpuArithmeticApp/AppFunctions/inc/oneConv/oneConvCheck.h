#pragma once
#ifndef CHECK_ONE_CONV
#define CHECK_ONE_CONV

#include "../maskAttributes/maskAttributes.h"

#include <cassert>
#include <iostream>
#include <vector>

void oneConvCheck(const int* mainVec, const int* maskVec, const int* resultVec, const int& conSize);

#endif