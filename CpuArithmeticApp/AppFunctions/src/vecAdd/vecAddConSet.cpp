#include "../../inc/vecAdd/vecAddConSet.h"


vecAddConSize vecAddConSet(vecAddConSize& n)
{
    int tempInput{0};

    bool correctChoice = false;

    do 
    {
        std::cout << "Please select vector addition element sample size from the options below:\n\n";
        std::cout << "25,000,000 elements:        enter '1'\n";
        std::cout << "35,000,000 elements:        enter '2'\n";
        std::cout << "45,000,000 elements:        enter '3'\n";
        std::cout << "55,000,000 elements:        enter '4'\n";
        std::cout << "65,000,000 elements:        enter '5'\n";

        std::cin.clear();
        std::cin >> tempInput;

        if (tempInput <= 0 || tempInput >= 6)
        {
            std::cout << "\nPlease select a valid option.\n\n";
        }
        else if (tempInput == 1)
        {
            n = 25000000; // 25 million elements
            correctChoice = true;
        }
        else if (tempInput == 2)
        {
            n = 35000000; // 35 million elements
            correctChoice = true;
        }
        else if (tempInput == 3)
        {
            n = 45000000; // 45 million elements
            correctChoice = true;
        }
        else if (tempInput == 4)
        {
            n = 55000000; // 55 million elements
            correctChoice = true;
        }
        else if (tempInput == 5)
        {
            n = 65000000; // 65 million elements
            correctChoice = true;
        }

    } while (correctChoice != true);

    return n;
}


