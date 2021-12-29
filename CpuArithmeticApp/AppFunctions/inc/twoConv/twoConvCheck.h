#pragma once
#ifndef CHECK_TWO_CONV
#define CHECK_TWO_CONV

// 7 x 7 convolutional mask
#define MASK_DIM 7
// Amount the the matrix will hang over the matrix
#define MASK_OFFSET (MASK_DIM / 2)

#include <iostream>
#include <vector>
#include "../allTDefs.h"

void twoConvCheck(std::vector<int> const&, std::vector<std::vector<int>> const&, std::vector<int> const&, twoConvConSize const&);

#endif