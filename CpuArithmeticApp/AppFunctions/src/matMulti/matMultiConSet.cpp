#include "../../inc/matMulti/matMultiConSet.h"


matMultiConSize matMultiConSet(matMultiConSize& n)
{

    int tempInput{ 0 };

    bool correctChoice = false;

    std::cout << "Please select matrix multiplication element sample size from the options below:\n";
    std::cout << "1. 1,000\n";
    std::cout << "2. 1,500\n";
    std::cout << "3. 2,000\n";
    std::cout << "4. 2,500\n";
    std::cout << "5. 3,000\n";

    do
    {
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

