#include "../../inc/core/opChoice.h"

void opChoice(int& userInput)
{

    bool correctChoice = false;

    do 
    {
        std::cout << "Please select an arithmetic operation from the options below:\n\n";
        std::cout << "Vector Addition:           enter '1'\n";
        std::cout << "Matrix Multiplication:     enter '2'\n";
        std::cout << "1D Convolution:            enter '3'\n";
        std::cout << "2D Convolution:            enter '4'\n\n";
        std::cout << "If you wish to close this program, please enter '5'\n";

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
                std::cout << "\n\nPlease select a valid option.\n\n";
            }
            else if (userInput == 1)
            {
                // load relevant vecAdd classes
                std::cout << "\nVector Addition has been selected.\n\n";
                correctChoice = true;
            }
            else if (userInput == 2)
            {
                // load relevant matMulti classes
                std::cout << "\nMatrix Multiplication has been selected.\n\n";
                correctChoice = true;
            }
            else if (userInput == 3)
            {
                // load relevant oneConv classes
                std::cout << "\n1D Convolution has been selected.\n\n";
                correctChoice = true;
            }
            else if (userInput == 4)
            {
                // load relevant twoConv classes
                std::cout << "\n2D Convolution has been selected.\n\n";
                correctChoice = true;
            }
            else if (userInput == 5)
            {
                bool closeChoice = false;

                std::cout << "\nAre you sure you want to close the program?\n";
                std::cout << "Yes (1) || No (2)\n";

                do
                {
                    std::cin.clear();

                    if (!(std::cin >> userInput))
                    {
                        std::cout << "\n\nPlease enter numbers only.\n\n";
                        std::cin.clear();
                        std::cin.ignore(std::numeric_limits<std::streamsize>::max(), '\n');
                    }
                    else if (userInput <= 0 || userInput >= 3)
                    {
                        std::cout << "\nNo correct option selected.\n\n";
                    }
                    else if (userInput == 1)
                    {
                        closeChoice = true;
                        correctChoice = true;
                        userInput = 5;
                    }
                    else if (userInput == 2)
                    {
                        std::cout << "\nReturning to arithmetic selection.\n\n";
                        closeChoice = true;
                    }

                } while (closeChoice != true);
            }
        }
    } while (correctChoice != true);
}