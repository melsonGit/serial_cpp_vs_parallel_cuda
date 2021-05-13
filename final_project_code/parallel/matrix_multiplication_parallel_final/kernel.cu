// Parallel Matrix Multiplication Program
//
// Code sourced and adpated from the following author/s and sources: 
// - https://github.com/CoffeeBeforeArch/from_scratch/blob/master/matrixMul/matrix_mul.cu
// - https://github.com/CoffeeBeforeArch/cuda_programming/blob/master/matrixMul/baseline/mmul.cu
// - https://thispointer.com/how-to-fill-a-vector-with-random-numbers-in-c/
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

__global__ void matrixMul(const int* a, const int* b, int* c, int no_elements) {
    // Compute each thread's global row and column index
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    // Iterate over row, and down column
    c[row * no_elements + col] = 0;
    for (int k = 0; k < no_elements; k++) {
        // Accumulate results for a single element
        c[row * no_elements + col] += a[row * no_elements + k] * b[k * no_elements + col];
    }
}

int main() {

    // Call element_set function to assign variable no_elements with a user selected value
    static int no_elements = element_set(no_elements);
    size_t bytes = no_elements * no_elements * sizeof(int);

    // Host vectors
    vector<int> h_a(no_elements * no_elements);
    vector<int> h_b(no_elements * no_elements);
    vector<int> h_c(no_elements * no_elements);

    // Start the clock
    clock_t start = clock();

    // Initialise vector matrices by generating random numbers via Lambda C++11 function
    generate(h_a.begin(), h_a.end(), []() {return rand() % 100;});
    generate(h_b.begin(), h_b.end(), []() {return rand() % 100;});

    // Allocate device memory
    int* d_a, * d_b, * d_c;
    cudaMalloc(&d_a, bytes);
    cudaMalloc(&d_b, bytes);
    cudaMalloc(&d_c, bytes);

    // Copy data to the device
    cudaMemcpy(d_a, h_a.data(), bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, h_b.data(), bytes, cudaMemcpyHostToDevice);

    // Threads per CTA dimension
    int THREADS = 32;

    // Blocks per grid dimension (assumes THREADS divides N evenly)
    int BLOCKS = no_elements / THREADS;

    // Use dim3 structs for block  and grid dimensions
    dim3 threads(THREADS, THREADS);
    dim3 blocks(BLOCKS, BLOCKS);

    // Launch kernel
    matrixMul << <blocks, threads >> > (d_a, d_b, d_c, no_elements);

    // Copy back to the host - Might be able to delete as this is for the Verify_result function
    cudaMemcpy(h_c.data(), d_c, bytes, cudaMemcpyDeviceToHost);

    // Free memory on device
    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c);

    clock_t end = clock();

    double diffs = (end - start) / (double)CLOCKS_PER_SEC;
    cout << diffs << "s Matrix Multiplication computation time, with an element size of " << no_elements << ".\n";
    cout << "PARALLEL MATRIX MULTIPLICATION COMPUTATION SUCCESSFUL.\nShutting down program....\n";

    return EXIT_SUCCESS;
}

// Function Declarations
int element_set(int element_size) {

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
    if (temp_input == 1) {
        element_size = 1000;
    }   // 1500 elements
    else if (temp_input == 2) {
        element_size = 1500;
    }   // 2000 elements
    else if (temp_input == 3) {
        element_size = 2000;
    }   // 2500 elements
    else if (temp_input == 4) {
        element_size = 2500;
    }   // 3000 elements
    else if (temp_input == 5) {
        element_size = 3000;
    }

    return element_size;
}