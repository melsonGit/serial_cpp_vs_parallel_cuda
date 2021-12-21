#include "../../inc/matMulti/matMultiNumGen.h"

void matMultiNumGen(std::vector<std::vector<int>>& a, std::vector<std::vector<int>>& b)
{
    std::cout << "\nMatrix Multiplication: Populating input vectors.\n";

    // Re-seed rand() function for each run
    srand((uint16_t)time(NULL));

    // Vector loop
    for (auto i = 0; i < a.size(); i++)
    {
        // Vector of vector of ints loop
        for (auto j = 0; j < a[i].size(); j++)
            //assign element random vector number elements of vector
            a[i][j] = rand() % 100;
    }

    for (auto k = 0; k < a.size(); k++)
    {
        //j loop
        for (auto l = 0; l < a[l].size(); l++)
            //assign element random vector number elements of vector
            b[k][l] = rand() % 100;
    }
    std::cout << "\nMatrix Multiplication: Populating complete.\n";
}