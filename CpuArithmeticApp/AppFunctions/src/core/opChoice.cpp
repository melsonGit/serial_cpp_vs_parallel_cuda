#include "../../inc/core/opChoice.h"

void opChoice()
{
    int tempInput;

    std::cout << "Please select an arithmetic operation from the options below:\n";
    std::cout << "1. Vector Addition\n";
    std::cout << "2. Matrix Multiplication\n";
    std::cout << "3. 1-D Convolution\n";
    std::cout << "4. 2-D Convolution\n";
    std::cin >> tempInput;

    if (tempInput <= 0 || tempInput >= 5)
    {
        // !###~~###! refactor to allow user another go !###~~###!
        std::cout << "\n\nNo correct option selected.\nShutting down program....\n";
        EXIT_FAILURE;
    }
    if (tempInput == 1) {
    } // load relevant vecAdd classes
    else if (tempInput == 2) {
    } // load relevant mMulti classes
    else if (tempInput == 3) {
    } // load relevant 1-D convo classes
    else if (tempInput == 4) {
      // load relevant 2-D convo classes
    }
}
