#include "../../inc/oneConv/oneConvFunc.h"

void oneConvFunc(std::vector<int> const& mainVec, std::vector<int> const& maskVec, std::vector<int>& resVec, 
                 oneConvConSize const& conSize, int const& maskSize)
{
    std::cout << "\n1D Convolution: Starting operation.\n";

    int radius{ maskSize / 2 };
    int oneConvOutput{ 0 };
    int start{ 0 };

    for (int i = 0; i < conSize; i++) {
        start = i - radius;
        oneConvOutput = 0;
        for (int j = 0; j < maskSize; j++) {
            if ((start + j >= 0) && (start + j < conSize)) {
                oneConvOutput += mainVec[start + j] * maskVec[j];
            }
        }
    }
    std::cout << "\n1D Convolution: Operation complete.\n";
}