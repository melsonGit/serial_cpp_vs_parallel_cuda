#include "../../inc/oneConv/oneConvConSet.h"

int oneConvConSet(int& conSize)
{

    int userInput { 0 };

    bool correctChoice { false };

    do
    {
        std::cout << "Please select 1D Convolution element sample size from the options below:\n\n";
        std::cout << "10,000,000 elements:        enter '1'\n";
        std::cout << "25,000,000 elements:        enter '2'\n";
        std::cout << "55,000,000 elements:        enter '3'\n";
        std::cout << "75,000,000 elements:        enter '4'\n";
        std::cout << "90,000,000 elements:        enter '5'\n";

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
                conSize = 10000000; // 10 million elements
                correctChoice = true;
            }
            else if (userInput == 2)
            {
                conSize = 25000000; // 25 million elements
                correctChoice = true;
            }
            else if (userInput == 3)
            {
                conSize = 55000000; // 55 million elements
                correctChoice = true;
            }
            else if (userInput == 4)
            {
                conSize = 75000000; // 75 million elements
                correctChoice = true;
            }
            else if (userInput == 5)
            {
                conSize = 90000000; // 90 million elements
                correctChoice = true;
            }
        }
    } while (!correctChoice);

    return conSize;
}