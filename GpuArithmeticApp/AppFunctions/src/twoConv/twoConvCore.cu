#include "../../inc/twoConv/twoConvCore.cuh"

void twoConvCore()
{
    // Initialise and allocate variable conSize with a user selected value
	int conSize { twoConvConSet(conSize) };

    // Initialise and allocate native arrays (hostMainVec; input) and (hostResultVec; output) a container size of conSize * conSize
    int* hostMainVec { new int[conSize * conSize] };
    int* hostResultVec { new int[conSize * conSize] };

    // Initialise and allocate the mask a container size of maskDim * maskDim
    int* hostMaskVec { new int[maskAttributes::maskDim * maskAttributes::maskDim] };

    // Populate input arrays
    std::cout << "\n2D Convolution: Populating main vector.\n";
    twoConvNumGen(hostMainVec, conSize);
    std::cout << "\n2D Convolution: Populating mask vector.\n";
    twoConvNumGen(hostMaskVec, maskAttributes::maskDim);

    std::cout << "\n2D Convolution: Populating complete.\n";

    // Initialise bytesVecMem/MaskMem to used for allocating memory to device vars
    // This allows us to copy data host to device and vice versa.
    size_t bytesVecMem { conSize * conSize * sizeof(int) };
    size_t bytesMaskMem { maskAttributes::maskDim * maskAttributes::maskDim * sizeof(int) };

    // Allocate memory on the device using cudaMalloc
    int* deviceMainVec, * deviceMaskVec, * deviceResultVec;
    cudaMalloc(&deviceMainVec, bytesVecMem);
    cudaMalloc(&deviceMaskVec, bytesMaskMem);
    cudaMalloc(&deviceResultVec, bytesVecMem);

    std::cout << "\n2D Convolution: Copying data from host to device.\n";

    // Copy data from the host to the device using cudaMemcpy | .data() returns pointer to memory used by vector/array to store its owned elements
    cudaMemcpy(deviceMainVec, hostMainVec, bytesVecMem, cudaMemcpyHostToDevice);
    cudaMemcpy(deviceMaskVec, hostMaskVec, bytesMaskMem, cudaMemcpyHostToDevice);

    // Threads per Cooperative Thread Array
    int THREADS { 32 };

    // No. CTAs per grid
    // Add padding | Enables compatibility with sample sizes not divisible by 32
    int BLOCKS { (conSize + THREADS - 1) / THREADS };

    // Use dim3 structs for BLOCKS and THREADS dimensions | Passed to kernal lauch as launch arguments
    dim3 threads(THREADS, THREADS);
    dim3 blocks(BLOCKS, BLOCKS);
    // Start the clock
    clock_t opStart { clock() };

    std::cout << "\n2D Convolution: Starting operation.\n";

    // Launch kernel on device
    twoConvFunc <<< blocks, threads >>> (deviceMainVec, deviceMaskVec, deviceResultVec, conSize);

    std::cout << "\n2D Convolution: Operation complete.\n";

    // Stop clock
    clock_t opEnd { clock() };

    std::cout << "\n2D Convolution: Copying results from device to host.\n";

    // Copy data from device back to host using cudaMemcpy
    cudaMemcpy(hostResultVec, deviceResultVec, bytesVecMem, cudaMemcpyDeviceToHost);

    std::cout << "\n2D Convolution: Copying complete.\n";

    // Authenticate results on host
    twoConvCheck(hostMainVec, hostMaskVec, hostResultVec, conSize);
    
    std::cout << "\n2D Convolution: Freeing host and device memory.\n\n";

    // Free allocated memory on the host and device
    delete[] hostMainVec;
    delete[] hostResultVec;
    delete[] hostMaskVec;

    cudaFree(deviceMainVec);
    cudaFree(deviceResultVec);

    // Calculate overall time spent to complete operation
    double completionTime { ((static_cast<double>(opEnd)) - (static_cast<double>(opStart))) / (double)CLOCKS_PER_SEC };

    // Output timing to complete operation and container size
    std::cout << completionTime << "s 2D Convolution computation time, with a container size of " << conSize * conSize << ".\n\n";
    std::cout << "Returning to selection screen.\n\n";

    std::cout << "#########################################################################\n" <<
                 "#########################################################################\n" <<
                 "#########################################################################\n\n";
}