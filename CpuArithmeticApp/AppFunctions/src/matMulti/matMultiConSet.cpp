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
            if (userInput <= 0 || userInput >= 6)
            {
                std::cout << "\nPlease select a valid option.\n\n";
            }
            else if (userInput == 1)
            {
                conSize = 524288;  // 524,288 X 2 = 1,048,576 elements
                correctChoice = true;
            }
            else if (userInput == 2)
            {
                conSize = 2097152;  // 2,097,152 X 2 = 4,194,304 elements
                correctChoice = true;
            }
            else if (userInput == 3)
            {
                conSize = 4718592;  // 4,718,592 X 2 = 9,437,184 elements
                correctChoice = true;
            }
            else if (userInput == 4)
            {
                conSize = 8388608;  // 8,388,608 X 2 = 16,777,216 elements
                correctChoice = true;
            }
            else if (userInput == 5)
            {
                conSize = 13107200; // 13,107,200 X 2 = 26,214,400 elements
                correctChoice = true;
            }
        }
    } while (!correctChoice);

    return conSize;
}

