#include "../../inc/vecAdd/vecConSet.h"


vecAddConSize conSet(vecAddConSize& n)
{
    int tempInput{0};

    bool correctChoice = false;

    std::cout << "Please select vector addition element sample size from the options below:\n";
    std::cout << "1. 25,000,000\n";
    std::cout << "2. 35,000,000\n";
    std::cout << "3. 45,000,000\n";
    std::cout << "4. 55,000,000\n";
    std::cout << "5. 65,000,000\n";

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


