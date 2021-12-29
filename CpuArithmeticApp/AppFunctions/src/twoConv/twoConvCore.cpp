#include "../../inc/twoConv/twoConvCore.h"

// 7 x 7 convolutional mask
#define MASK_DIM 7

// Amount the the matrix will hang over the matrix
#define MASK_OFFSET (MASK_DIM / 2)


void twoConvCore()
{
    // Assign variable conSize with a user selected value
    twoConvConSize conSize = twoConvConSet(conSize);

    // Assign main vector with size of conSize * conSize
    std::vector<int> mainVec(conSize * conSize);
    // Assign mask vector with size of MASK_DIM * MASK_DIM
    std::vector<int> maskVec(MASK_DIM * MASK_DIM);




}

int main() {




    clock_t start = clock();

    init_matrix(main_vector, no_elements);
    init_matrix(h_mask, MASK_DIM);

    convolution_2d(main_vector, h_mask, no_elements);

    clock_t end = clock();

    double diffs = (end - start) / (double)CLOCKS_PER_SEC;
    cout << diffs << "s 2-D Convolution computation time, with an element size of " << no_elements << ".\n";
    cout << "SEQUENTIAL 2-D CONVOLUTION COMPUTATION SUCCESSFUL.\nShutting down program....\n";

    return EXIT_SUCCESS;
}


// Executes 2D convolution
// Arguments:
//  main_vector = Original matrix
//  mask = Convolutional mask
//  no_elements = Dimensions of the matrix

void convolution_2d(vector<int> main_vector, vector<int> mask, int no_elements) {
    // Variable for accumulating results
    int convo_output;

    // Intermediate value for more readable code
    int offset_r;
    int offset_c;

    // Go over each row
    for (int i = 0; i < no_elements; i++) {
        // Go over each column
        for (int j = 0; j < no_elements; j++) {
            // Assign the convo_output variable a value
            convo_output = 0;

            // Go over each mask row
            for (int k = 0; k < MASK_DIM; k++) {
                // Update offset value for row
                offset_r = i - MASK_OFFSET + k;

                // Go over each mask column
                for (int l = 0; l < MASK_DIM; l++) {
                    // Update offset value for column
                    offset_c = j - MASK_OFFSET + l;

                    // Range checks if hanging off the matrix
                    if (offset_r >= 0 && offset_r < no_elements) {
                        if (offset_c >= 0 && offset_c < no_elements) {
                            // Accumulate results into convo_output
                            convo_output += main_vector[offset_r * no_elements + offset_c] * mask[k * MASK_DIM + l];
                        }
                    }
                }
            }
        }
    }
}

// Initialises matrix
// Arguments:
//  main_vector = Matrix
//  no_elements = Dimension of the matrix (square)
void init_matrix(vector<int> main_vector, int no_elements) {
    for (int i = 0; i < no_elements; i++) {
        for (int j = 0; j < no_elements; j++) {
            main_vector[no_elements * i + j] = rand() % 100;
        }
    }
}