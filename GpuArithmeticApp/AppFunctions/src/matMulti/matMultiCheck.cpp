#include "../../inc/matMulti/matMultiCheck.h"

void matMultiCheck(std::vector<int> const& inputA, std::vector<int> const& inputB, std::vector<int> const& outputVec, const int& conSize) {

    std::cout << "\nMatrix Multiplication: Authenticating results.\n\n";

    bool doesMatch { true };

    // For each row
    for (auto rowIn { 0 }; rowIn < conSize && doesMatch; ++rowIn) 
    {
        // For every column...
        for (auto colIn { 0 }; colIn < conSize; ++colIn) 
        {
            // For every element in the row-column pair
            int resultVar { 0 };

            for (auto rowColPair { 0 }; rowColPair < conSize; ++rowColPair) 
            {
                // Accumulate the partial results
                resultVar += inputA[rowIn * conSize + rowColPair] * inputB[rowColPair * conSize + colIn];
            }

            if (resultVar != outputVec[rowIn * conSize + colIn])
                doesMatch = false;
            else
                continue;
        }
    }

    if (!doesMatch)
        std::cout << "Matrix multiplication unsuccessful: output vector data does not match expected results.\n"
        << "Timing results will be discarded.\n";
    else
        std::cout << "Matrix multiplication successful: output vector data matches expected results.\n"
        << "Timing results will be recorded.\n";
}