#include "../../inc/vecAdd/conSet.h"


vecAddConSize conSet(vecAddConSize& n)
{
    userChoice tempInput;

    std::cout << "Please select vector addition element sample size from the options below:\n";
    std::cout << "1. 25,000,000\n";
    std::cout << "2. 35,000,000\n";
    std::cout << "3. 45,000,000\n";
    std::cout << "4. 55,000,000\n";
    std::cout << "5. 65,000,000\n";
    std::cin >> tempInput;

    if (tempInput <= 0 || tempInput >= 6)
    {
        std::cout << "\n\nNo correct option selected!\nShutting down program....\n";
        EXIT_FAILURE;
    }
    
    if (tempInput == 1) 
    {
        n = 25000000; // 25 million elements
    } 
    else if (tempInput == 2) 
    {
        n = 35000000; // 35 million elements
    } 
    else if (tempInput == 3) 
    {
        n = 45000000; // 45 million elements
    } 
    else if (tempInput == 4) 
    {
        n = 55000000; // 55 million elements
    } 
    else if (tempInput == 5) 
    {
        n = 65000000; // 65 million elements
    }

    return n;
}


