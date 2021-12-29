#pragma once
#ifndef TWO_CONV_FUNC
#define TWO_CONV_FUNC

// 7 x 7 convolutional mask
#define MASK_DIM 7
// Amount the the matrix will hang over the matrix
#define MASK_OFFSET (MASK_DIM / 2)

#include <iostream>
#include <vector>
#include "../allTDefs.h"

void twoConvFunc(std::vector<int> const&, std::vector<std::vector<int>> const&, std::vector<int>&, twoConvConSize const&);

#endif