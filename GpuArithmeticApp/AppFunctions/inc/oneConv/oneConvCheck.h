#pragma once
#ifndef CHECK_ONE_CONV
#define CHECK_ONE_CONV

#ifndef MASK_ONE_DIM
// Number of elements in the convolution mask
#define MASK_ONE_DIM 7
#endif

#include <iostream>
#include <vector>

void oneConvCheck(int const*, int const*, int const*, int const&);

#endif