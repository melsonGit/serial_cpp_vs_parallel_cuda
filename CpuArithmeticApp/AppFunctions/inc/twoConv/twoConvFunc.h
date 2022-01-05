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

void twoConvFunc(std::vector<int> const&, std::vector<int> const&, std::vector<int>&, int const&);

#endif