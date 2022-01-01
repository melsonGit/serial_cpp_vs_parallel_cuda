#include "../../inc/twoConv/twoConvConSet.h"

twoConvConSize twoConvConSet (twoConvConSize& n) 
{
    int userInput { 0 };

    bool correctChoice { false };

    do
    {
        std::cout << "Please select 2D Convolution element sample size from the options below:\n\n";
        std::cout << "4,096  elements:        enter '1'\n";
        std::cout << "5,120  elements:        enter '2'\n";
        std::cout << "6.144  elements:        enter '3'\n";
        std::cout << "8,192  elements:        enter '4'\n";
        std::cout << "10,240 elements:        enter '5'\n";

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
                n = 4096; // 4,096 elements
                correctChoice = true;
            }
            else if (userInput == 2)
            {
                n = 5120; // 5,120 elements
                correctChoice = true;
            }
            else if (userInput == 3)
            {
                n = 6144; // 6,144 elements
                correctChoice = true;
            }
            else if (userInput == 4)
            {
                n = 8192; // 8,192 elements
                correctChoice = true;
            }
            else if (userInput == 5)
            {
                n = 10240; // 10,240 elements
                correctChoice = true;
            }
        }
    } while (!correctChoice);

    return n;
}


