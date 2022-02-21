#include "../../inc/vecAdd/vecAddConSet.h"

int vecAddConSet(int& conSize)
{
    int userInput { 0 };

    bool correctChoice { false };

    do
    {
        std::cout << "Please select vector addition element sample size from the options below:\n\n";
        std::cout << "25,000,000 elements:        enter '1'\n";
        std::cout << "35,000,000 elements:        enter '2'\n";
        std::cout << "45,000,000 elements:        enter '3'\n";
        std::cout << "55,000,000 elements:        enter '4'\n";
        std::cout << "65,000,000 elements:        enter '5'\n";

        std::cin.clear();

        if (!(std::cin >> userInput))
        {
            std::cout << "\nPlease enter numbers only.\n\n";
            std::cin.clear();
            std::cin.ignore(std::numeric_limits<std::streamsize>::max(), '\n');
        }
        else
        {
            switch (userInput)
            {
            case 1:
                conSize = 25000000; // 25 million elements
                correctChoice = true;
                break;
            case 2:
                conSize = 35000000; // 35 million elements
                correctChoice = true;
                break;
            case 3:
                conSize = 45000000; // 45 million elements
                correctChoice = true;
                break;
            case 4:
                conSize = 55000000; // 55 million elements
                correctChoice = true;
                break;
            case 5:
                conSize = 65000000; // 65 million elements
                correctChoice = true;
                break;
            default:
                std::cout << "\nPlease select a valid option.\n\n";
                break;
            }
        }
    } while (!correctChoice);

    return conSize;
}