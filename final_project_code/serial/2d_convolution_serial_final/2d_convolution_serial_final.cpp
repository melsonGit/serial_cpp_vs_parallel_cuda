/* Sources:
- https://www.youtube.com/watch?v=qxcfco89wvs
- https://github.com/CoffeeBeforeArch/cuda_programming/blob/master/convolution/2d_constant_memory/convolution.cu
- https://mathworld.wolfram.com/Convolution.html
*/

#include <cassert>
#include <cstdlib>
#include <iostream>
#include <vector>

using std::cout;
using std::cin;
using std::vector;

// 7 x 7 convolutional mask
#define MASK_DIM 7

// Amount the the matrix will hang over the matrix
#define MASK_OFFSET (MASK_DIM / 2)

// Allocate mask in constant memory
int mask[7 * 7];

// Function Prototypes
int element_set(int);
void init_matrix(vector<int>, int);
void convolution_2d(vector<int>, vector<int>, int);

int main() {


    // Call element_set function to assign variable no_elements with a user selected value || Sets number of elements to be used
    static int no_elements = element_set(no_elements);

    clock_t start = clock();

    // Allocate the matrix and initialize it
    vector<int> matrix(no_elements * no_elements);
    init_matrix(matrix, no_elements);

    // Allocate the mask and initialize it
    vector<int> h_mask (MASK_DIM * MASK_DIM);
    init_matrix(h_mask, MASK_DIM);

    convolution_2d(matrix, h_mask, no_elements);

    clock_t end = clock();

    double diffs = (end - start) / (double)CLOCKS_PER_SEC;
    cout << diffs << "s 2-D Convolution computation time, with an element size of " << no_elements << ".\n";
    cout << "SEQUENTIAL 2-D CONVOLUTION COMPUTATION SUCCESSFUL.\nShutting down program....\n";

    return EXIT_SUCCESS;
}

int element_set(int element_size) {

    int temp_input;

    cout << "Please select 2-D convolution element sample size from the options below:\n";
    cout << "1. 5,120\n";
    cout << "2. 10,240\n";
    cout << "3. 15,360\n";
    cout << "4. 20,480\n";
    cout << "5. 25,600\n";
    cin >> temp_input;

    if (temp_input <= 0 || temp_input >= 6)
    {
        cout << "\n\nNo correct option selected!\nShutting down program....\n";
        return EXIT_FAILURE;
    }
    // Work from a bases of 1024 and times that to get a good array number
    // 5120 elements
    if (temp_input == 1) {
        element_size = 5120;
    } // 10240 elements
    else if (temp_input == 2) {
        element_size = 10240; // Appears to be the max for SERIAL
    } // 15360 elements
    else if (temp_input == 3) {
        element_size = 15360;
    } // 20480 elements
    else if (temp_input == 4) {
        element_size = 20480;
    } // 25600 elements
    else if (temp_input == 5) {
        element_size = 25600;
    }

    return element_size;
}

// Initializes an n x n matrix with random numbers
// Takes:
//  m : Pointer to the matrix
//  no_elements : Dimension of the matrix (square)
void init_matrix(vector<int> m, int no_elements) {
    for (int i = 0; i < no_elements; i++) {
        for (int j = 0; j < no_elements; j++) {
            m[no_elements * i + j] = rand() % 100;
        }
    }
}

// Verifies the 2D convolution result on the CPU
// Takes:
//  m:      Original matrix
//  mask:   Convolutional mask
//  no_elements:      Dimensions of the matrix

void convolution_2d(vector<int> m, vector<int> mask, int no_elements) {
    // Temp value for accumulating results
    int temp;

    // Intermediate value for more readable code
    int offset_r;
    int offset_c;

    // Go over each row
    for (int i = 0; i < no_elements; i++) {
        // Go over each column
        for (int j = 0; j < no_elements; j++) {
            // Reset the temp variable
            temp = 0;

            // Go over each mask row
            for (int k = 0; k < MASK_DIM; k++) {
                // Update offset value for row
                offset_r = i - MASK_OFFSET + k;

                // Go over each mask column
                for (int l = 0; l < MASK_DIM; l++) {
                    // Update offset value for column
                    offset_c = j - MASK_OFFSET + l;

                    // Range checks if we are hanging off the matrix
                    if (offset_r >= 0 && offset_r < no_elements) {
                        if (offset_c >= 0 && offset_c < no_elements) {
                            // Accumulate partial results
                            temp += m[offset_r * no_elements + offset_c] * mask[k * MASK_DIM + l];
                        }
                    }
                }
            }
        }
    }
}


