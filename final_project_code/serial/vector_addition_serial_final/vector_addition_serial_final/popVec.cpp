#include "popVec.h"

popVec::popVec(int x)
{
    elementSize += x;
}

int popVec::elementSet(int& elementSize)
{
    int tempInput;

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
    // 25 million elements
    if (tempInput == 1) {
        elementSize = 25000000;
    } // 35 million elements
    else if (tempInput == 2) {
        elementSize = 35000000;
    } // 45 million elements
    else if (tempInput == 3) {
        elementSize = 45000000;
    } // 55 million elements
    else if (tempInput == 4) {
        elementSize = 55000000;
    } // 65 million elements
    else if (tempInput == 5) {
        elementSize = 65000000;
    }

    return elementSize;
}


