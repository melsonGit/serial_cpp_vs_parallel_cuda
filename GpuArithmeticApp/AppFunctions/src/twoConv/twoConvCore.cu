#include "../../inc/twoConv/twoConvCore.cuh"

// Number of elements in the convolution mask
#ifndef MASK_TWO_DIM
#define MASK_TWO_DIM 7
#endif

// Allocate mask in constant memory
__constant__ int maskConstant[7 * 7];

void twoConvCore()
{
	// Assign variable conSize with a user selected value
	int conSize { twoConvConSet(conSize) };

    // Size of the matrix in bytes
    size_t bytesVecMem { conSize * conSize * sizeof(int) };

    // Size of the mask in bytes
    size_t bytesMaskMem = MASK_TWO_DIM * MASK_TWO_DIM * sizeof(int);

    // Allocate the matrix and initialise it
    int* hostMainVec{ new int[conSize * conSize] };
    int* hostResVec{ new int[conSize * conSize] };

    // Allocate the mask and initialise it
    int* hostMaskVec { new int[MASK_TWO_DIM * MASK_TWO_DIM] };

    std::cout << "\n2D Convolution: Populating main vector.\n";
    twoConvNumGen(hostMainVec, conSize);
    std::cout << "\n2D Convolution: Populating mask vector.\n";
    twoConvNumGen(hostMaskVec, MASK_TWO_DIM);

    std::cout << "\n2D Convolution: Populating complete.\n";

    // Allocate device memory
    int* deviceMainVec, * deviceResVec;
    cudaMalloc(&deviceMainVec, bytesVecMem);
    cudaMalloc(&deviceResVec, bytesVecMem);

    std::cout << "\n2D Convolution: Copying data from host to device.\n";

    // Copy data to the device
    cudaMemcpy(deviceMainVec, hostMainVec, bytesVecMem, cudaMemcpyHostToDevice);
    cudaMemcpyToSymbol(maskConstant, hostMaskVec, bytesMaskMem);

    // Calculate grid dimensions + block buffer to avoid off by one errors
    int THREADS { 16 };
    int BLOCKS { (conSize + THREADS - 1) / THREADS };

    // Dimension launch arguments
    dim3 threads(THREADS, THREADS);
    dim3 blocks(BLOCKS, BLOCKS);

    clock_t opStart{ clock() };

    std::cout << "\n2D Convolution: Starting operation.\n";

    // Launch 2D convolution kernel
    twoConvFunc <<< blocks, threads >>> (deviceMainVec, deviceResVec, conSize);

    std::cout << "\n2D Convolution: Operation complete.\n";

    clock_t opEnd{ clock() };

    std::cout << "\n2D Convolution: Copying results from device to host.\n";

    // Copy the result back to the CPU
    cudaMemcpy(hostResVec, deviceResVec, bytesVecMem, cudaMemcpyDeviceToHost);

    twoConvCheck(hostMainVec, hostMaskVec, hostResVec, conSize);
    
    std::cout << "\n2D Convolution: Freeing host and device memory.\n\n";

    // Free allocated memory on the host and device
    delete[] hostMainVec;
    delete[] hostResVec;
    delete[] hostMaskVec;

    cudaFree(deviceMainVec);
    cudaFree(deviceResVec);

    // Calculate overall time spent to complete operation
    double completionTime{ (opEnd - opStart) / (double)CLOCKS_PER_SEC };

    // Output timing to complete operation and container size
    std::cout << completionTime << "s 2D Convolution computation time, with a container size of " << conSize * conSize << ".\n\n";
    std::cout << "Returning to selection screen.\n\n";

    std::cout << "#########################################################################\n" <<
                 "#########################################################################\n" <<
                 "#########################################################################\n\n";
}