#include "../../inc/matMulti/matMultiConSet.h"

int matMultiConSet(int& n)
{

    int userInput { 0 };

    bool correctChoice { false };

    do
    {
        std::cout << "Please select matrix multiplication element sample size from the options below:\n\n";
        std::cout << "2,000 elements:        enter '1'\n";
        std::cout << "3,000 elements:        enter '2'\n";
        std::cout << "4,000 elements:        enter '3'\n";
        std::cout << "5,000 elements:        enter '4'\n";
        std::cout << "6,000 elements:        enter '5'\n";

        std::cin.clear();

        if (!(std::cin >> userInput))
        {
            std::cout << "\nPlease enter numbers only.\n\n";
            std::cin.clear();
            std::cin.ignore(std::numeric_limits<std::streamsize>::max(), '\n');
        }
        else 
        {
            if (userInput <= 0 || userInput >= 6)
            {
                std::cout << "\nPlease select a valid option.\n\n";
            }
            else if (userInput == 1)
            {
                n = 1000; // 1,000 x 2 matrix
                correctChoice = true;
            }
            else if (userInput == 2)
            {
                n = 1500; // 1500 x 2 matrix
                correctChoice = true;
            }
            else if (userInput == 3)
            {
                n = 2000; // 2000 x 2 matrix
                correctChoice = true;
            }
            else if (userInput == 4)
            {
                n = 2500; // 2500 x 2 matrix
                correctChoice = true;
            }
            else if (userInput == 5)
            {
                n = 3000; // 3000 x 2 matrix
                correctChoice = true;
            }
        }
    } while (!correctChoice);

    return n;
}

