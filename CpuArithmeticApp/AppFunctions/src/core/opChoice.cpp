#include "../../inc/core/opChoice.h"

void opChoice(int& userInput)
{
    bool correctChoice { false };

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
            switch (userInput)
            {
            case 1:
                std::cout << "\nVector Addition has been selected.\n\n";
                correctChoice = true;
                break;
            case 2:
                std::cout << "\nMatrix Multiplication has been selected.\n\n";
                correctChoice = true;
                break;
            case 3:
                std::cout << "\n1D Convolution has been selected.\n\n";
                correctChoice = true;
                break;
            case 4:
                std::cout << "\n2D Convolution has been selected.\n\n";
                correctChoice = true;
                break;
            case 5:
            {
                bool closeChoice { false };

                do
                {
                    std::cout << "\nAre you sure you want to close the program?\n";
                    std::cout << "Yes (1) || No (2)\n";

                    std::cin.clear();

                    if (!(std::cin >> userInput))
                    {
                        std::cout << "\n\nPlease enter numbers only.\n\n";
                        std::cin.clear();
                        std::cin.ignore(std::numeric_limits<std::streamsize>::max(), '\n');
                    }
                    else
                    {
                        switch (userInput)
                        {
                        case 1:
                            closeChoice = true;
                            correctChoice = true;
                            userInput = 5;
                            std::cout << "\nClosing program.\n";
                            break;
                        case 2:
                            std::cout << "\nReturning to arithmetic selection.\n\n";
                            closeChoice = true;
                            break;
                        default:
                            std::cout << "\nPlease select a valid option.\n\n";
                            break;
                        }
                    }
                } while (!closeChoice);
            }
            break;
            default:
                std::cout << "\n\nPlease select a valid option.\n\n";
                break;
            }
        }
    } while (!correctChoice);
}