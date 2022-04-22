#pragma once
#ifndef TWO_CONV_NUM_GEN
#define TWO_CONV_NUM_GEN

#include "../maskAttributes/maskAttributes.h"
#include "../randNumGen/randNumGen.h"

#include <algorithm>
#include <cstdlib>
#include <time.h>

void twoConvNumGen(int* vecToPop, const int& conSize);

#endif