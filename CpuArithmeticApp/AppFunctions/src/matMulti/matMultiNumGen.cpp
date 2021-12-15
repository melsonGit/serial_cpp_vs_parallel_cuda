#include "../../inc/matMulti/matMultiNumGen.h"

// Function Prototypes
#if 0
int element_set(int);
void matrix_multi(vector<int>, vector<int>, vector<int>, int);

int main() {

    // Call element_set function to assign variable no_elements with a user selected value
    static int no_elements = element_set(no_elements);

    // Initialise vector size
    vector<int> a(no_elements * no_elements), b(no_elements * no_elements), c(no_elements * no_elements);

    // Start the clock
    clock_t start = clock();

    // Initialise vector matrices by generating random numbers via Lambda C++11 function
    generate(a.begin(), a.end(), []() {return rand() % 100; });
    generate(b.begin(), b.end(), []() {return rand() % 100; });

    matrix_multi(a, b, c, no_elements);


    clock_t end = clock();

    double diffs = (end - start) / (double)CLOCKS_PER_SEC;
    cout << diffs << "s Matrix Multiplication computation time, with an element size of " << no_elements << ".\n";
    cout << "SEQUENTIAL MATRIX MULTIPLICATION COMPUTATION SUCCESSFUL.\nShutting down program....\n";

    return EXIT_SUCCESS;
}
#endif