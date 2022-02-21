#include "../../inc/matMulti/matMultiCheck.h"

void matMultiCheck(std::vector<int> const& inputVecA, std::vector<int> const& inputVecB, std::vector<int> const& resultVec, const int& conSize) 
{
    std::cout << "\nMatrix Multiplication: Authenticating results.\n\n";

    // Determines result authenticity - Assigned false value when results don't match
    bool doesMatch { true };

    // Accumulates our results to check against resultVec
    int resultVar{};

    // For each row
    for (auto rowId { 0 }; rowId < conSize && doesMatch; ++rowId) 
    {
        // For each column in that row
        for (auto colId { 0 }; colId < conSize; ++colId) 
        {
            // Reset resultVar to 0 on next element
            resultVar = 0;

            // For every element in the row-column pair
            for (auto rowColPairId { 0 }; rowColPairId < conSize; ++rowColPairId) 
            {
                // Accumulate results into resultVar
                resultVar += inputVecA[rowId * conSize + rowColPairId] * inputVecB[rowColPairId * conSize + colId];
            }
            // Check accumulated resultVar value with corresponding value in resultVec
            if (resultVar != resultVec[rowId * conSize + colId])
                doesMatch = false;
        }
    }
    // Assert and abort when results don't match
    assert(doesMatch && "Check failed! Accumulated resultVar value doesn't match corresponding value in resultVec (matMulti).");

    if (!doesMatch)
        std::cerr << "Matrix multiplication unsuccessful: output vector data does not match expected results.\n"
        << "Timing results will be discarded.\n";
    else
        std::cout << "Matrix multiplication successful: output vector data matches expected results.\n"
        << "Timing results will be recorded.\n";
}