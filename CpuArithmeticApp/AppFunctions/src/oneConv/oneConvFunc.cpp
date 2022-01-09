#include "../../inc/oneConv/oneConvFunc.h"

void oneConvFunc(std::vector<int> const& mainVec, std::vector<int> const& maskVec, std::vector<int>& resVec, 
                 int const& conSize)
{
    std::cout << "\n1D Convolution: Populating complete.\n";
    std::cout << "\n1D Convolution: Starting operation.\n";

    // Radius will determine when convolution occurs to prevent out of bound errors
    int maskRadius { MASK_ONE_DIM / 2 };
    int start { 0 };

    for (auto i { 0 }; i < conSize; i++)
    {
        start = i - maskRadius;
        resVec[i] = 0;

        for (auto j { 0 }; j < MASK_ONE_DIM; j++)
        {
            if ((start + j >= 0) && (start + j < conSize)) 
            {
                resVec[i] += mainVec[start + j] * maskVec[j];
            }
        }
    }
    std::cout << "\n1D Convolution: Operation complete.\n";
}