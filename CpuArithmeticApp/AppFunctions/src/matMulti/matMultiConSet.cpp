#include "../../inc/matMulti/matMultiConSet.h"


matMultiConSize matMultiConSet(matMultiConSize& n)
{

    int tempInput{ 0 };

    bool correctChoice = false;

    do
    {
        std::cout << "Please select matrix multiplication element sample size from the options below:\n\n";
        std::cout << "1,000 elements:        enter '1'\n";
        std::cout << "1,500 elements:        enter '2'\n";
        std::cout << "2,000 elements:        enter '3'\n";
        std::cout << "2,500 elements:        enter '4'\n";
        std::cout << "3,000 elements:        enter '5'\n";

        std::cin.clear();
        std::cin >> tempInput;

        if (tempInput <= 0 || tempInput >= 6)
        {
            std::cout << "\nPlease select a valid option.\n\n";
        }
        else if (tempInput == 1)
        {
            n = 1000; // 1,000 elements
            correctChoice = true;
        }
        else if (tempInput == 2)
        {
            n = 1500; // 1500 elements
            correctChoice = true;
        }
        else if (tempInput == 3)
        {
            n = 2000; // 2000 elements
            correctChoice = true;
        }
        else if (tempInput == 4)
        {
            n = 2500; // 2500 elements
            correctChoice = true;
        }
        else if (tempInput == 5)
        {
            n = 3000; // 3000 elements
            correctChoice = true;
        }

    } while (correctChoice != true);

    return n;
}

