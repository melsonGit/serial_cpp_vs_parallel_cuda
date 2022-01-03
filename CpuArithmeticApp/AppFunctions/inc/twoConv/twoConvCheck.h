#pragma once
#ifndef CHECK_TWO_CONV
#define CHECK_TWO_CONV

#ifndef MASK_TWO_DIM
// 7 x 7 convolutional mask
#define MASK_TWO_DIM 7
#endif

#ifndef MASK_OFFSET
// Amount the the matrix will hang over the matrix
#define MASK_OFFSET (MASK_TWO_DIM / 2)
#endif

#include <iostream>
#include <vector>
#include "../allTDefs.h"

void twoConvCheck(std::vector<int> const&, std::vector<int> const&, std::vector<int> const&, twoConvConSize const&);

#endif