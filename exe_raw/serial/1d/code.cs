// Sequential Naive 1-D Convolution Program
//
// Code sourced and adpated from the following author/s and sources: 
// - https://www.youtube.com/watch?v=OlLquh9Lnbc
// - https://github.com/CoffeeBeforeArch/cuda_programming/blob/6589c89a78dee44e14ccb362cdae69f2e6850a2c/convolution/1d_naive/convolution.cu
// - https://mathworld.wolfram.com/Convolution.html
// Please refer to the bibliography for a complete reference of the above author/s and sources


#include <algorithm>
#include <iostream>
#include <vector>

using std::cout;
using std::cin;
using std::generate;
using std::vector;

// Function Prototypes
int element_set(int);
void convolution(vector<int>, vector<int>, int, int);

int main()
{

    // Call element_set function to assign variable no_elements with a user selected value || Sets number of elements to be used
    static int no_elements = element_set(no_elements);

    // Number of elements in the convolution mask
    int m = 7;

    // Allocate the vector with no_elements
    vector<int> main_vector(no_elements);
    // Allocate the mask with m
    vector<int> mask(m);

    clock_t start = clock();

    // Generate random numbers via Lambda C++11 function, and place into vector
    generate(begin(main_vector), end(main_vector), []() { return rand() % 100; });
// initialise mask || m mumber of elements in vector are randomised between 1 - 10
generate(begin(mask), end(mask), []() { return rand() % 10; });

convolution(main_vector, mask, no_elements, m);

clock_t end = clock();

double diffs = (end - start) / (double)CLOCKS_PER_SEC;
cout << diffs << "s 1-D Naive Convolution computation time, with an element size of " << no_elements << ".\n";
cout << "SEQUENTIAL 1-D NAIVE CONVOLUTION COMPUTATION SUCCESSFUL.\nShutting down program....\n";

return EXIT_SUCCESS;
}


int element_set(int element_size)
{

    int temp_input;

    cout << "Please select 1-D naive convolution element sample size from the options below:\n";
    cout << "1. 10,000,000\n";
    cout << "2. 25,000,000\n";
    cout << "3. 55,000,000\n";
    cout << "4. 75,000,000\n";
    cout << "5. 90,000,000\n";
    cin >> temp_input;

    if (temp_input <= 0 || temp_input >= 6)
    {
        cout << "\n\nNo correct option selected!\nShutting down program....\n";
        return EXIT_FAILURE;
    }
    // 10 million elements
    if (temp_input == 1)
    {
        element_size = 10000000;
    } // 25 million elements
    else if (temp_input == 2)
    {
        element_size = 25000000;
    } // 55 million elements
    else if (temp_input == 3)
    {
        element_size = 55000000;
    } // 75 million elements
    else if (temp_input == 4)
    {
        element_size = 75000000;
    } // 90 million elements
    else if (temp_input == 5)
    {
        element_size = 90000000;
    }

    return element_size;
}

void convolution(vector<int> main_vector, vector<int> mask, int element_size, int m)
{

    int radius = m / 2;
    int convo_output;
    int start;

    for (int i = 0; i < element_size; i++)
    {
        start = i - radius;
        convo_output = 0;
        for (int j = 0; j < m; j++)
        {
            if ((start + j >= 0) && (start + j < element_size))
            {
                convo_output += main_vector[start + j] * mask[j];
            }
        }
    }
}






