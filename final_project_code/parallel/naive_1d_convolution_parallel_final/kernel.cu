// Parallel Naive 1-D Convolution Program
//
// Code sourced and adpated from the following author/s and sources: 
// - https://www.youtube.com/watch?v=OlLquh9Lnbc
// - https://github.com/CoffeeBeforeArch/cuda_programming/blob/master/convolution/1d_naive/convolution.cu
// - https://mathworld.wolfram.com/Convolution.html
// Please refer to the bibliography for a complete reference of the above author/s and sources


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

__global__ void convolution_1d(int* array, int* mask, int* result, int no_elements,
    int m) {
    // Global thread ID calculation
    int tid = blockIdx.x * blockDim.x + threadIdx.x;

    // Calculate radius of the mask
    int r = m / 2;

    // Calculate the starting point for the element
    int start = tid - r;

    // Temp value for calculation
    int temp = 0;

    // Go over each element of the mask
    for (int j = 0; j < m; j++) {
        // Ignore elements that hang off (0s don't contribute)
        if (((start + j) >= 0) && (start + j < no_elements)) {
            // accumulate partial results
            temp += array[start + j] * mask[j];
        }
    }

    // Write-back the results
    result[tid] = temp;
}

// Function Prototypes
int element_set(int);

int main() {

    // Call element_set function to assign variable no_elements with a user selected value || Sets number of elements to be used
    static int no_elements = element_set(no_elements);

    // Size of the vector in bytes
    int bytes_n = no_elements * sizeof(int);

    // Number of elements in the convolution mask
    int m = 7;

    // Size of mask in bytes
    int bytes_m = m * sizeof(int);

    // Allocate the vector with no_elements...
    vector<int> h_array(no_elements);

    clock_t start = clock();

    // Generate random numbers via Lambda C++11 function, and place into vector
    generate(begin(h_array), end(h_array), []() { return rand() % 100; });

    // Allocate the mask and initialise it || m mumber of elements in vector are randomised between 1 - 10
    vector<int> h_mask(m);
    generate(begin(h_mask), end(h_mask), []() { return rand() % 10; });

    // Allocate space for the result
    vector<int> h_result(no_elements);

    // Allocate space on the device
    int* d_array, * d_mask, * d_result;
    cudaMalloc(&d_array, bytes_n);
    cudaMalloc(&d_mask, bytes_m);
    cudaMalloc(&d_result, bytes_n);

    // Copy the data to the device
    cudaMemcpy(d_array, h_array.data(), bytes_n, cudaMemcpyHostToDevice);
    cudaMemcpy(d_mask, h_mask.data(), bytes_m, cudaMemcpyHostToDevice);

    // Threads per TB (thread blocks)
    int THREADS = 256;

    // Number of TBs
    int GRID = (no_elements + THREADS - 1) / THREADS;

    // Call the kernel
    convolution_1d << <GRID, THREADS >> > (d_array, d_mask, d_result, no_elements, m);

    // Copy back to the host
    cudaMemcpy(h_result.data(), d_result, bytes_n, cudaMemcpyDeviceToHost);

    // Free allocated memory on the device and host
    cudaFree(d_result);
    cudaFree(d_mask);
    cudaFree(d_array);

    clock_t end = clock();

    double diffs = (end - start) / (double)CLOCKS_PER_SEC;
    cout << diffs << "s 1-D Naive Convolution computation time, with an element size of " << no_elements << ".\n";
    cout << "PARALLEL 1-D NAIVE CONVOLUTION COMPUTATION SUCCESSFUL.\nShutting down program....\n";

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
    }   // 25 million elements
    else if (temp_input == 2) {
        element_size = 25000000;
    }   // 55 million elements
    else if (temp_input == 3) {
        element_size = 55000000;
    }   // 75 million elements
    else if (temp_input == 4) {
        element_size = 75000000;
    }   // 90 million elements
    else if (temp_input == 5) {
        element_size = 90000000;
    }

    return element_size;
}