#include "../../inc/twoConv/twoConvNumGen.h"

void twoConvNumGen(std::vector<int>& mainVec, std::vector<std::vector<int>>& maskVec)
{
    std::cout << "\n2D Convolution: Populating main and mask vectors.\n";

    // Re-seed rand() function for each run
    srand((unsigned int)time(NULL));

    // Generate random numbers via Lambda C++11 function, and place into vector
    std::generate(mainVec.begin(), mainVec.end(), []() { return rand() % 100; });

    // Loop to populate 2D vector maskVec
    // For each row
    for (auto iRow = 0; iRow < maskVec.size(); iRow++)
    {
        // For each column in that row
        for (auto iCol = 0; iCol < maskVec[iRow].size(); iCol++)
        {
            // Assign random number to vector of vector of ints to columns iCol of rows iRows
            maskVec[iRow][iCol] = rand() % 100; // ##!! CHANGE TO 10 IF ISSUES ARISE!!##
        }
    }

    std::cout << "\n2D Convolution: Populating complete.\n";
}