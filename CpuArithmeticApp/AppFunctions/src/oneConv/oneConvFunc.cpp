#include "../../inc/oneConv/oneConvFunc.h"

void oneConvFunc(std::vector<int> const& mainVec, std::vector<int> const& maskVec, std::vector<int>& resVec, 
                 oneConvConSize const& conSize)
{
    std::cout << "\n1D Convolution: Starting operation.\n";

    int radius{ MASK_ONE_DIM / 2 };
    int start{ 0 };

    for (int i = 0; i < conSize; i++) {
        start = i - radius;
        resVec[i] = 0;
        for (int j = 0; j < MASK_ONE_DIM; j++) {
            if ((start + j >= 0) && (start + j < conSize)) {
                resVec[i] += mainVec[start + j] * maskVec[j];
            }
        }
    }
    std::cout << "\n1D Convolution: Operation complete.\n";
}