// Sequential Matrix Multiplication Program 
//
// Code sourced and adpated from the following author/s and sources:
// - https://www.programiz.com/cpp-programming/examples/matrix-multiplication
// - https://github.com/CoffeeBeforeArch/cuda_programming/blob/6589c89a78dee44e14ccb362cdae69f2e6850a2c/matrixMul/baseline/mmul.cu
// - https://docs.microsoft.com/en-us/archive/msdn-magazine/2012/april/c-amp-introduction-to-tiling-in-c-amp
// - https://docs.microsoft.com/en-us/cpp/parallel/amp/walkthrough-matrix-multiplication?view=msvc-160
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
void matrix_multi(vector<int>, vector<int>, vector<int>, int);

int main()
{

    // Call element_set function to assign variable no_elements with a user selected value
    static int no_elements = element_set(no_elements);

    // Initialise vector size
    vector<int> a(no_elements* no_elements), b(no_elements * no_elements), c(no_elements * no_elements);

    // Start the clock
    clock_t start = clock();

    // Initialise vector matrices by generating random numbers via Lambda C++11 function
    generate(a.begin(), a.end(), []() { return rand() % 100; });
generate(b.begin(), b.end(), []() { return rand() % 100; });

matrix_multi(a, b, c, no_elements);


clock_t end = clock();

double diffs = (end - start) / (double)CLOCKS_PER_SEC;
cout << diffs << "s Matrix Multiplication computation time, with an element size of " << no_elements << ".\n";
cout << "SEQUENTIAL MATRIX MULTIPLICATION COMPUTATION SUCCESSFUL.\nShutting down program....\n";

return EXIT_SUCCESS;
}

// Function Declarations
int element_set(int element_size)
{

    int temp_input;

    cout << "Please select matrix multiplication element sample size from the options below:\n";
    cout << "1. 1,000\n";
    cout << "2. 1,500\n";
    cout << "3. 2,000\n";
    cout << "4. 2,500\n";
    cout << "5. 3,000\n";
    cin >> temp_input;

    if (temp_input <= 0 || temp_input >= 6)
    {
        cout << "\n\nNo correct option selected!\nShutting down program....\n";
        return EXIT_FAILURE;
    }
    // 1000 elements
    if (temp_input == 1)
    {
        element_size = 1000;
    }   // 1500 elements
    else if (temp_input == 2)
    {
        element_size = 1500;
    }   // 2000 elements
    else if (temp_input == 3)
    {
        element_size = 2000;
    }   // 2500 elements
    else if (temp_input == 4)
    {
        element_size = 2500;
    }   // 3000 elements
    else if (temp_input == 5)
    {
        element_size = 3000;
    }

    return element_size;
}

void matrix_multi(vector<int> matrix_a, vector<int> matrix_b, vector<int> matrix_mult, int elements)
{
    // For each row
    for (int row = 0; row < elements; row++)
    {
        // For each column
        for (int col = 0; col < elements; col++)
        {
            // For every elements in the row-column couple
            matrix_mult[row * elements + col] = 0;
            for (int k = 0; k < elements; k++)
            {
                // Store results of a single element from matrix_a and matrix_b into a single element of matrix_mult
                matrix_mult[row * elements + col] += matrix_a[row * elements + k] * matrix_b[k * elements + col];
            }
        }
    }
}