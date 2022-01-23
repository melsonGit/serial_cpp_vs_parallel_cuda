#pragma once
#ifndef CHECK_MULTI
#define CHECK_MULTI

#include <iostream>
#include <vector>

void matMultiCheck(std::vector<std::vector<int>> const& inputVecA, std::vector<std::vector<int>> const& inputVecB, 
				   std::vector<std::vector<int>> const& resultVec, const int& numRows);

#endif