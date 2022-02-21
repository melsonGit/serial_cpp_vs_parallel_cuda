#include "../../inc/twoConv/twoConvConSet.h"

int twoConvConSet(int& conSize)
{
    int userInput { 0 };

    bool correctChoice { false };

    do
    {
        std::cout << "Please select 2D Convolution element sample size from the options below:\n\n";
        std::cout << "16,777,216  elements:        enter '1'\n";
        std::cout << "26,214,400  elements:        enter '2'\n";
        std::cout << "37,748,736  elements:        enter '3'\n";
        std::cout << "67,108,864  elements:        enter '4'\n";
        std::cout << "104,857,600 elements:        enter '5'\n";

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
                conSize = 4096; // 4,096 x 4,096 = 16,777,216 elements
                correctChoice = true;
                break;
            case 2:
                conSize = 5120; // 5,120 x 5,120 = 26,214,400 elements
                correctChoice = true;
                break;
            case 3:
                conSize = 6144; // 6,144 x 6,144 = 37,748,736 elements
                correctChoice = true;
                break;
            case 4:
                conSize = 8192; // 8,192 x 8,192 = 67,108,864 elements
                correctChoice = true;
                break;
            case 5:
                conSize = 10240; // 10,240 x 10,240 = 104,857,600 elements
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