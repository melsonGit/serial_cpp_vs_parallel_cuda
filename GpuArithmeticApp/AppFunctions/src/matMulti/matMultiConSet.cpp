#include "../../inc/matMulti/matMultiConSet.h"

int matMultiConSet(int& conSize)
{
    int userInput { 0 };

    bool correctChoice { false };

    do
    {
        std::cout << "Please select matrix multiplication element sample size from the options below:\n\n";
        std::cout << "1,048,576  elements:        enter '1'\n";
        std::cout << "4,194,304  elements:        enter '2'\n";
        std::cout << "9,437,184  elements:        enter '3'\n";
        std::cout << "16,777,216 elements:        enter '4'\n";
        std::cout << "26,214,400 elements:        enter '5'\n";

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
                conSize = 1024;  // 1,024 x 1,024 = 1,048,576 elements
                correctChoice = true;
                break;
            case 2:
                conSize = 2048;  // 2,048 x 2,048 = 4,194,304 elements
                correctChoice = true;
                break;
            case 3:
                conSize = 3072;  // 3,072 x 3,072 = 9,437,184 elements
                correctChoice = true;
                break;
            case 4:
                conSize = 4096;  // 4,096 x 4,096 = 16,777,216 elements
                correctChoice = true;
                break;
            case 5:
                conSize = 5120;  // 5,120 x 5,120 = 26,214,400 elements
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