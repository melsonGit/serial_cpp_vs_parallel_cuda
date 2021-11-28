// Parallel 2-D Convolution Program
//
// Code sourced and adpated from the following author/s and sources: 
// - https://www.youtube.com/watch?v=qxcfco89wvs
// - https://github.com/CoffeeBeforeArch/cuda_programming/blob/6589c89a78dee44e14ccb362cdae69f2e6850a2c/convolution/2d_constant_memory/convolution.cu
// - https://mathworld.wolfram.com/Convolution.html
// Please refer to the bibliography for a complete reference of the above author/s and sources

#include <iostream>
#include <cstdlib>

using std::cout;
using std::cin;

// 7 x 7 convolutional mask
#define MASK_DIM 7

// Amount the the matrix will hang over the matrix
#define MASK_OFFSET (MASK_DIM / 2)

// Allocate mask in constant memory
__constant__ int mask[7 * 7];

// 2D Convolution Kernel
// Takes:
//  matrix: Input matrix
//  result: Convolution result
//  no_elements:      Dimensions of the matrices
__global__ void convolution_2d(int* matrix, int* result, int no_elements)
{
    // Calculate the global thread positions
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    // Starting index for calculation
    int start_r = row - MASK_OFFSET;
    int start_c = col - MASK_OFFSET;

    // Temp value for accumulating the result
    int temp = 0;

    // Iterate over all the rows
    for (int i = 0; i < MASK_DIM; i++)
    {
        // Go over each column
        for (int j = 0; j < MASK_DIM; j++)
        {
            // Range check for rows
            if ((start_r + i) >= 0 && (start_r + i) < no_elements)
            {
                // Range check for columns
                if ((start_c + j) >= 0 && (start_c + j) < no_elements)
                {
                    // Accumulate result
                    temp += matrix[(start_r + i) * no_elements + (start_c + j)] *
                        mask[i * MASK_DIM + j];
                }
            }
        }
    }

    // Write back the result
    result[row * no_elements + col] = temp;
}

// Function Prototype
int element_set(int);
void init_matrix(int*, int);


int main()
{


    // Call element_set function to assign variable no_elements with a user selected value || Sets number of elements to be used
    static int no_elements = element_set(no_elements);

    // Size of the matrix (in bytes)
    size_t bytes_n = no_elements * no_elements * sizeof(int);

    // Size of the mask in bytes
    size_t bytes_m = MASK_DIM * MASK_DIM * sizeof(int);

    // Allocate the mask and initialise it
    int* h_mask = new int[MASK_DIM * MASK_DIM];

    // Allocate the matrix and initialise it
    int* h_matrix = new int[no_elements * no_elements];
    int* h_result = new int[no_elements * no_elements];

    clock_t start = clock();

    init_matrix(h_matrix, no_elements);
    init_matrix(h_mask, MASK_DIM);

    // Allocate device memory
    int* d_matrix;
    int* d_result;
    cudaMalloc(&d_matrix, bytes_n);
    cudaMalloc(&d_result, bytes_n);

    // Copy data to the device
    cudaMemcpy(d_matrix, h_matrix, bytes_n, cudaMemcpyHostToDevice);
    cudaMemcpyToSymbol(mask, h_mask, bytes_m);

    // Calculate grid dimensions
    int THREADS = 16;
    int BLOCKS = (no_elements + THREADS - 1) / THREADS;

    // Dimension launch arguments
    dim3 threads(THREADS, THREADS);
    dim3 blocks(BLOCKS, BLOCKS);

    // Perform 2D Convolution
    convolution_2d << < blocks, threads >> > (d_matrix, d_result, no_elements);

    // Copy the result back to the CPU
    cudaMemcpy(h_result, d_result, bytes_n, cudaMemcpyDeviceToHost);

    // Free allocated memory on the device and host
    cudaFree(h_matrix);
    cudaFree(h_result);
    cudaFree(h_mask);
    cudaFree(d_matrix);
    cudaFree(d_result);

    clock_t end = clock();

    double diffs = (end - start) / (double)CLOCKS_PER_SEC;
    cout << diffs << "s 2-D Convolution computation time, with an element size of " << no_elements << ".\n";
    cout << "PARALLEL 2-D CONVOLUTION COMPUTATION SUCCESSFUL.\nShutting down program....\n";

    return EXIT_SUCCESS;

}

// Function Declarations
int element_set(int element_size)
{

    int temp_input;

    cout << "Please select 2-D convolution element sample size from the options below:\n";
    cout << "1. 4,096\n";
    cout << "2. 5,120\n";
    cout << "3. 6,144\n";
    cout << "4. 8,192\n";
    cout << "5. 10,240\n";
    cin >> temp_input;

    if (temp_input <= 0 || temp_input >= 6)
    {
        cout << "\n\nNo correct option selected!\nShutting down program....\n";
        return EXIT_FAILURE;
    }
    // 4096 elements
    if (temp_input == 1)
    {
        element_size = 4096;
    }   // 5120 elements
    else if (temp_input == 2)
    {
        element_size = 5120;
    }   // 6144 elements
    else if (temp_input == 3)
    {
        element_size = 6144;
    }   // 8192 elements
    else if (temp_input == 4)
    {
        element_size = 8192;
    }   // 10240 elements
    else if (temp_input == 5)
    {
        element_size = 10240;
    }

    return element_size;
}

// Initialises a matrix
// Takes:
//  m = Pointer to the matrix
//  no_elements = Dimension of the matrix (square)
void init_matrix(int* m, int no_elements)
{
    for (int i = 0; i < no_elements; i++)
    {
        for (int j = 0; j < no_elements; j++)
        {
            m[no_elements * i + j] = rand() % 100;
        }
    }
}

