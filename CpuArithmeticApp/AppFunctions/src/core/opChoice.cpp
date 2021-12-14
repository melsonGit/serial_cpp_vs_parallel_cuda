#include "../../inc/core/opChoice.h"

void opChoice(int& input)
{

    bool correctSelect = false;

    do 
    {
        std::cout << "Please select an arithmetic operation from the options below:\n";
        std::cout << "1. Vector Addition: enter '1'\n";
        std::cout << "2. Matrix Multiplication: enter '2'\n";
        std::cout << "3. 1-D Convolution: enter '3'\n";
        std::cout << "4. 2-D Convolution: enter '4'\n";
        std::cout << "If you wish to close this program, please enter '5'\n";
        std::cin.clear();
        std::cin >> input;

        if (input <= 0 || input >= 6)
        {
            std::cout << "\nNo correct option selected.\n";
        }
        else if (input == 1)
        {
            // load relevant vecAdd classes
            std::cout << "\nVector Addition has been selected.\n";
            correctSelect = true;
        }
        else if (input == 2)
        {
            // load relevant matMulti classes
            std::cout << "\nMatrix Multiplication has been selected.\n";
            correctSelect = true;
        }
        else if (input == 3)
        {
            // load relevant 1-D convo classes
            std::cout << "\n1-D Convolution has been selected.\n";
            correctSelect = true;
        }
        else if (input == 4)
        {
            // load relevant 2-D convo classes
            std::cout << "\n2-D Convolution has been selected.\n";
            correctSelect = true;
        }
        else if (input == 5)
        {
            bool closeChoice = false;

            do
            {

                std::cout << "\nAre you sure you want to close the program?\n";
                std::cout << "Yes (1) || No (2)\n";
                std::cin.clear();
                std::cin >> input;

                if (input <= 0 || input >= 3)
                {
                    std::cout << "\nNo correct option selected.\n";
                }
                else if (input == 1)
                {
                    closeChoice = true;
                    correctSelect = true;
                    input = 5;
                }
                else if (input == 2)
                {
                    std::cout << "\nReturning to arithmetic selection.\n\n";
                    closeChoice = true;
                }

            } while (closeChoice == false);
        }

    } while (correctSelect == false);
}