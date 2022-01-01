#include "../../inc/twoConv/twoConvFunc.h"

void twoConvFunc(std::vector<int> const& mainVec, std::vector<std::vector<int>> const& maskVec, std::vector<int>& resVec, 
                 twoConvConSize const& conSize)
{

    // Intermediate value for more readable code
    int offset_r;
    int offset_c;

    // Go over each row
    for (int i = 0; i < conSize; i++)
    {
        // Go over each column
        for (int j = 0; j < conSize; j++)
        {
            // Assign the resVec variable a value
            resVec[i] = 0;

            // Go over each mask row
            for (int k = 0; k < MASK_TWO_DIM; k++) 
            {
                // Update offset value for row
                offset_r = i - MASK_OFFSET + k;

                // Go over each mask column
                for (int l = 0; l < MASK_TWO_DIM; l++) 
                {
                    // Update offset value for column
                    offset_c = j - MASK_OFFSET + l;

                    // Range checks if hanging off the matrix
                    if (offset_r >= 0 && offset_r < conSize)
                    {
                        if (offset_c >= 0 && offset_c < conSize)
                        {
                            // Accumulate results into resVec
                            resVec[i] += mainVec[offset_r * conSize + offset_c] * maskVec[k * MASK_TWO_DIM][l];
                        }
                    }
                }
            }
        }
    }
}