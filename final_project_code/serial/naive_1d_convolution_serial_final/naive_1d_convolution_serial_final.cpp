/* Sources:
- https://www.youtube.com/watch?v=OlLquh9Lnbc
- https://github.com/CoffeeBeforeArch/cuda_programming/blob/master/convolution/1d_naive/convolution.cu
- https://mathworld.wolfram.com/Convolution.html
*/


#include <algorithm>
#include <cassert>
#include <cstdlib>
#include <iostream>
#include <vector>

using std::cout;
using std::cin;
using std::generate;
using std::vector;

// 1-D convolution kernel
//  Arguments:
//      array   = padded array
//      mask    = convolution mask
//      result  = result array
//      no_elements       = number of elements in array
//      m       = number of elements in the mask

// Function Prototypes
int element_set(int);
void convolution(vector<int>, vector<int>, vector<int>, int, int);

int main() {

    // Call element_set function to assign variable no_elements with a user selected value || Sets number of elements to be used
    static int no_elements = element_set(no_elements);

    // Size of the array in bytes
    int bytes_n = no_elements * sizeof(int);

    // Number of elements in the convolution mask
    int m = 7;

    // Allocate the vector with no_elements...
    vector<int> h_array(no_elements);

    clock_t start = clock();

    // Generate random numbers via Lambda C++11 function, and place into vector
    generate(begin(h_array), end(h_array), []() { return rand() % 100; });

    // Allocate the mask and initialize it || m mumber of elements in vector are randomised between 1 - 10
    vector<int> h_mask(m);
    generate(begin(h_mask), end(h_mask), []() { return rand() % 10; });

    // Allocate space for the result
    vector<int> h_result(no_elements);

    convolution(h_array, h_mask, h_result, no_elements, m);

    clock_t end = clock();

    double diffs = (end - start) / (double)CLOCKS_PER_SEC;
    cout << diffs << "s 1-D Naive Convolution computation time, with an element size of " << no_elements << ".\n";
    cout << "SEQUENTIAL 1-D NAIVE CONVOLUTION COMPUTATION SUCCESSFUL.\nShutting down program....\n";

    return EXIT_SUCCESS;
}

// Function Prototypes
int element_set(int element_size) {

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
    if (temp_input == 1) {
        element_size = 10000000;
    } // 25 million elements
    else if (temp_input == 2) {
        element_size = 25000000;
    } // 55 million elements
    else if (temp_input == 3) {
        element_size = 55000000;
    } // 75 million elements
    else if (temp_input == 4) {
        element_size = 75000000;
    } // 90 million elements
    else if (temp_input == 5) {
        element_size = 90000000;
    }

    return element_size;
}

void convolution(vector<int> array, vector<int> mask, vector<int> result, int element_size, int m) {

    int radius = m / 2;
    int temp;
    int start;

    for (int i = 0; i < element_size; i++) {
        start = i - radius;
        temp = 0;
        for (int j = 0; j < m; j++) {
            if ((start + j >= 0) && (start + j < element_size)) {
                temp += array[start + j] * mask[j];
            }
        }
    }
}
