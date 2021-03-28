// Parallel Vector Addition Program
/*
 Code sourced and adpated from the following author/s and sourcess:
- https://solarianprogrammer.com/2012/04/11/vector-addition-benchmark-c-cpp-fortran/
- https://github.com/CoffeeBeforeArch/cuda_programming/blob/master/vectorAdd/baseline/vectorAdd.cu
- https://www.youtube.com/watch?v=QVVTsLmMlwk&t
- https://thispointer.com/how-to-fill-a-vector-with-random-numbers-in-c/
 Please refer to the bibliography for a complete reference of the above author/s and sources
*/

#include <algorithm>
#include <iostream>
#include <vector>

using std::cout;
using std::cin;
using std::generate;
using std::vector;

// Function Prototypes
int element_set(int);

// Kernal that will be called from host (CPU) and run on the device (GPU)
// Function that computes the sum of two arrays
// CUDA kernel for vector addition || __global__ means this is called from the CPU, and runs on the GPU
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
    vector<int> a;
    vector<int> b;
    vector<int> c;

    a.reserve(no_elements);
    b.reserve(no_elements);
    c.reserve(no_elements);

    // Initialise vector by generating random numbers via Lambda C++11 function
    generate(a.begin(), a.end(), []() {
        return rand() % 100;
        });
    generate(b.begin(), b.end(), []() {
        return rand() % 100;
        });

    // Alternative but slower || Initialize random numbers in each array
    //for (int i = 0; i < no_elements; i++) {
    //    a.push_back(rand() % 100);
    //    b.push_back(rand() % 100);
    //}

    // Allocate memory on the device (GPU)
    int* d_a, * d_b, * d_c;
    cudaMalloc(&d_a, bytes);
    cudaMalloc(&d_b, bytes);
    cudaMalloc(&d_c, bytes);

    // Start the clock
    clock_t start = clock();

    // Copy data from the host to the device (CPU -> GPU)
    cudaMemcpy(d_a, a.data(), bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, b.data(), bytes, cudaMemcpyHostToDevice);

    // Threads per Cooperative Thread Array (CTA; 1024)
    int NUM_THREADS = 1 << 10;

    // CTAs per Grid
    // We need to launch at LEAST as many threads as we have elements
    // This equation pads an extra CTA to the grid if N cannot evenly be divided
    // by NUM_THREADS (e.g. N = 1025, NUM_THREADS = 1024)
    int NUM_BLOCKS = (no_elements + NUM_THREADS - 1) / NUM_THREADS;

    // Launch the kernel on the GPU
    // Kernel calls are asynchronous (the CPU program continues execution after
    // call, but not necessarily before the kernel finishes)
    vectorAdd << <NUM_BLOCKS, NUM_THREADS >> > (d_a, d_b, d_c, no_elements);

    // Copy sum vector from device to host
    // cudaMemcpy is a synchronous operation, and waits for the prior kernel
    // launch to complete (both go to the default stream in this case).
    // Therefore, this cudaMemcpy acts as both a memcpy and synchronization
    // barrier.
    cudaMemcpy(c.data(), d_c, bytes, cudaMemcpyDeviceToHost);

    // Stop the clock just after the vectorAdd function finishes executing
    clock_t end = clock();

    double diffs = (end - start) / (double)CLOCKS_PER_SEC;
    cout << diffs << "s Vector Addition computation time, with an element size of " << no_elements << ".\n";
    cout << "PARALLEL VECTOR ADDITION COMPUTATION SUCCESSFUL.\nShutting down program....\n";

    // Free memory on device
    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c);

    return EXIT_SUCCESS;
}

// Function Declarations
int element_set(int element_size) {

    int temp_input;

    cout << "Please select vector addition element sample size from the options below:\n";
    cout << "1. 55,000,000\n";
    cout << "2. 100,000,000\n";
    cout << "3. 150,000,000\n";
    cout << "4. 200,000,000\n";
    cout << "5. 280,000,000\n";
    cin >> temp_input;

    if (temp_input <= 0 || temp_input >= 6)
    {
        cout << "\n\nNo correct option selected!\nShutting down program....\n";
        return EXIT_FAILURE;
    }

    if (temp_input == 1) {
        element_size = 55000000;
    }
    else if (temp_input == 2) {
        element_size = 100000000;
    }
    else if (temp_input == 3) {
        element_size = 150000000;
    }
    else if (temp_input == 4) {
        element_size = 200000000;
    }
    else if (temp_input == 5) {
        element_size = 280000000;
    }

    return element_size;
}
