#include "../../inc/oneConv/oneConvFunc.h"
#include "../../inc/maskAttributes.h"

void oneConvFunc(std::vector<int> const& mainVec, std::vector<int> const& maskVec, std::vector<int>& resVec, const int& conSize)
{
    std::cout << "\n1D Convolution: Populating complete.\n";
    std::cout << "\n1D Convolution: Starting operation.\n";

    int start { 0 };

    for (auto i { 0 }; i < conSize; ++i)
    {
        start = i - maskAttributes::maskOffset;
        resVec[i] = 0;

        for (auto j { 0 }; j < maskAttributes::maskDim; ++j)
        {
            if ((start + j >= 0) && (start + j < conSize)) 
            {
                resVec[i] += mainVec[start + j] * maskVec[j];
            }
        }
    }
    std::cout << "\n1D Convolution: Operation complete.\n";
}