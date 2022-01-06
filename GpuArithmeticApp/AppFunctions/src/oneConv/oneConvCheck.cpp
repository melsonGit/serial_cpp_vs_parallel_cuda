#include "../../inc/oneConv/oneConvCheck.h"

void oneConvCheck(const int* mainVec, const int* maskVec, const int* resVec, const int& conSize)
{
    std::cout << "\n1D Convolution: Authenticating results.\n\n";

    int maskRadius { MASK_ONE_DIM / 2 };
    int startPoint { 0 };
    int resultVar;

    bool doesMatch { true };

    for (auto i { 0 }; i < conSize && doesMatch; i++)
    {
        startPoint = i - maskRadius;
        resultVar = 0;

        for (auto j { 0 }; j < MASK_ONE_DIM; j++)
        {
            if ((startPoint + j >= 0) && (startPoint + j < conSize)) 
            {
                resultVar += mainVec[startPoint + j] * maskVec[j];
            }
        }

        if (resultVar != resVec[i])
            doesMatch = false;
        else
            continue;
    }
    if (!doesMatch)
        std::cout << "1D Convolution unsuccessful: output vector data does not match the expected result.\n"
        << "Timing results will be discarded.\n\n";
    else
        std::cout << "1D Convolution successful: output vector data matches expected results.\n"
        << "Timing results will be recorded.\n\n";
}