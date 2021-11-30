// Parallel Vector Addition Program

#include <algorithm>
#include <iostream>
#include <vector>

using std::cout;
using std::cin;
using std::generate;
using std::vector;

// Function Prototypes
int element_set(int);

// CUDA kernel for vector addition || Function that computes the sum of two vectors
__global__ void vectorAdd(const int* __restrict a, const int* __restrict b,
    int* __restrict c, int no_elements) {
    // Calculate global thread ID
    int tid = (blockIdx.x * blockDim.x) + threadIdx.x;

    // Boundary check
    if (tid < no_elements) c[tid] = a[tid] + b[tid];

}

int main() {

    // Call element_set function to assign variable no_elements with a user selected value
    static int no_elements = element_set(no_elements);
    size_t bytes = sizeof(int) * no_elements;

    // Vectors for holding the host-side (CPU-side) data
    vector<int> a (no_elements);
    vector<int> b (no_elements);
    vector<int> c (no_elements);

    // Start the clock
    clock_t start = clock();

    // Initialise vector by generating random numbers via Lambda C++11 function
    generate(a.begin(), a.end(), []() {return rand() % 100;});
    generate(b.begin(), b.end(), []() {return rand() % 100;});

    // Allocate memory on the device (GPU)
    int* d_a, * d_b, * d_c;
    cudaMalloc(&d_a, bytes);
    cudaMalloc(&d_b, bytes);
    cudaMalloc(&d_c, bytes);

    // Copy data from the host to the device (CPU -> GPU)
    cudaMemcpy(d_a, a.data(), bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, b.data(), bytes, cudaMemcpyHostToDevice);

    // Threads per Cooperative Thread Array (CTA; 1024)
    int NUM_THREADS = 1 << 10;

    // CTAs per Grid
    int NUM_BLOCKS = (no_elements + NUM_THREADS - 1) / NUM_THREADS;

    // Launch the kernel on the GPU
    vectorAdd << <NUM_BLOCKS, NUM_THREADS >> > (d_a, d_b, d_c, no_elements);

    // Copy sum vector from device to host
    cudaMemcpy(c.data(), d_c, bytes, cudaMemcpyDeviceToHost);

    // Free memory on device
    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c);
    
    clock_t end = clock();

    double diffs = (end - start) / (double)CLOCKS_PER_SEC;
    cout << diffs << "s Vector Addition computation time, with an element size of " << no_elements << ".\n";
    cout << "PARALLEL VECTOR ADDITION COMPUTATION SUCCESSFUL.\nShutting down program....\n";

    return EXIT_SUCCESS;
}

// Function Declarations
int element_set(int element_size) {

    int temp_input;

    cout << "Please select vector addition element sample size from the options below:\n";
    cout << "1. 25,000,000\n";
    cout << "2. 35,000,000\n";
    cout << "3. 45,000,000\n";
    cout << "4. 55,000,000\n";
    cout << "5. 65,000,000\n";
    cin >> temp_input;

    if (temp_input <= 0 || temp_input >= 6)
    {
        cout << "\n\nNo correct option selected!\nShutting down program....\n";
        return EXIT_FAILURE;
    }
        // 25 million elements
    if (temp_input == 1) {
        element_size = 25000000;
    }   // 35 million elements
    else if (temp_input == 2) {
        element_size = 35000000;
    }   // 45 million elements
    else if (temp_input == 3) {
        element_size = 45000000;
    }   // 55 million elements
    else if (temp_input == 4) {
        element_size = 55000000;
    }   // 65 million elements
    else if (temp_input == 5) {
        element_size = 65000000;
    }

    return element_size;
}

