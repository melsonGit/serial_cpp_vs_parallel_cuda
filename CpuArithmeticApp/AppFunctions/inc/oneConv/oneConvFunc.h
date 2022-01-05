#pragma once
#ifndef ONE_CONV_FUNC
#define ONE_CONV_FUNC

#ifndef MASK_ONE_DIM
// Number of elements in the convolution mask
#define MASK_ONE_DIM 7
#endif

#include <iostream>
#include <vector>

void oneConvFunc(std::vector<int> const&, std::vector<int> const&, std::vector<int>&, int const&);

#endif