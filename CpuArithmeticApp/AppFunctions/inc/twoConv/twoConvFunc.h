#pragma once
#ifndef TWO_CONV_FUNC
#define TWO_CONV_FUNC

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

void twoConvFunc(std::vector<std::vector<int>> const&, std::vector<std::vector<int>> const&, std::vector<std::vector<int>>&, twoConvConSize const&);

#endif