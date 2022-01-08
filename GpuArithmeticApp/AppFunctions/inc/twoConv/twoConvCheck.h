#pragma once
#ifndef CHECK_TWO_CONV
#define CHECK_TWO_CONV

#ifndef MASK_TWO_DIM
// Number of elements in the convolution mask
#define MASK_TWO_DIM 7
#endif

#include <iostream>

void twoConvCheck(const int* mainVec, const int* maskVec, const int* resVec, const int& conSize);

#endif
