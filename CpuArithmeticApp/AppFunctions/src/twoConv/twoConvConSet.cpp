#include "../../inc/twoConv/twoConvConSet.h"

int twoConvConSet (int& conSize) 
{
    int userInput { 0 };

    bool correctChoice { false };

    do
    {
        std::cout << "Please select 2D Convolution element sample size from the options below:\n\n";
        std::cout << "4,096  elements:        enter '1'\n";
        std::cout << "5,184  elements:        enter '2'\n";
        std::cout << "6,400  elements:        enter '3'\n";
        std::cout << "8,836  elements:        enter '4'\n";
        std::cout << "10,201 elements:        enter '5'\n";

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
                conSize = 64; // 64 x 64 = 4,096 elements
                correctChoice = true;
            }
            else if (userInput == 2)
            {
                conSize = 72; // 72 x 72 = 5,184 elements
                correctChoice = true;
            }
            else if (userInput == 3)
            {
                conSize = 80; // 80 x 80 = 6,400 elements
                correctChoice = true;
            }
            else if (userInput == 4)
            {
                conSize = 94; // 94 x 94 = 8,836 elements
                correctChoice = true;
            }
            else if (userInput == 5)
            {
                conSize = 101; // 101 x 101 = 10,201 elements
                correctChoice = true;
            }
        }
    } while (!correctChoice);

    return conSize;
}


