#include "../../inc/oneConv/oneConvCore.cuh"

using Clock = std::chrono::steady_clock;

void oneConvCore()
{
	// Initialise and allocate variable conSize with a user selected value
	int conSize { oneConvConSet(conSize) };

	// Initialise and allocate main vector and resultant vector with size conSize
	std::vector<int> hostMainVec(conSize), hostResVec(conSize);

	// Initialise and allocate mask vector with maskDim
	std::vector<int> hostMaskVec(maskAttributes::maskDim);

	// Popluate input vectors
	std::cout << "\n1D Convolution: Populating main vector.\n";
	oneConvNumGen(hostMainVec);
	std::cout << "\n1D Convolution: Populating mask vector.\n";
	oneConvNumGen(hostMaskVec);

	std::cout << "\n1D Convolution: Populating complete.\n";

	// Initialise bytesVecMem/MaskMem to used for allocating memory to device vars
	// This allows us to copy data host to device and vice versa.
	size_t bytesVecMem { conSize * sizeof(int) };
	size_t bytesMaskMem { maskAttributes::maskDim * sizeof(int) };

	// Allocate memory on the device using cudaMalloc
	int* deviceMainVec, * deviceMaskVec, * deviceResVec;
	cudaMalloc(&deviceMainVec, bytesVecMem);
	cudaMalloc(&deviceMaskVec, bytesMaskMem);
	cudaMalloc(&deviceResVec, bytesVecMem);

	std::cout << "\n1D Convolution: Copying data from host to device.\n";

	// Copy data from the host to the device using cudaMemcpy | .data() returns pointer to memory used by vector/array to store its owned elements
	cudaMemcpy(deviceMainVec, hostMainVec.data(), bytesVecMem, cudaMemcpyHostToDevice);
	cudaMemcpy(deviceMaskVec, hostMaskVec.data(), bytesMaskMem, cudaMemcpyHostToDevice);

	// Threads per Cooperative Thread Array
	int THREADS { 32 };

	// No. CTAs per grid
	// Add padding | Enables compatibility with sample sizes not divisible by 32
	int BLOCKS { (conSize + THREADS - 1) / THREADS };

	std::cout << "\n1D Convolution: Starting operation.\n";

	// Start clock
	auto opStart { Clock::now() };

	// Launch kernel on device
	oneConvFunc << <BLOCKS, THREADS >> > (deviceMainVec, deviceMaskVec, deviceResVec, conSize);

	// Stop clock
	auto opEnd { Clock::now() };

	std::cout << "\n1D Convolution: Operation complete.\n";
	std::cout << "\n1D Convolution: Copying results from device to host.\n";

	// Copy data from device back to host using cudaMemcpy
	cudaMemcpy(hostResVec.data(), deviceResVec, bytesVecMem, cudaMemcpyDeviceToHost);

	std::cout << "\n2D Convolution: Copying complete.\n";

	// Authenticate results on host
	oneConvCheck(hostMainVec.data(), hostMaskVec.data(), hostResVec.data(), conSize);

	std::cout << "\n1D Convolution: Freeing device memory.\n\n";

	// Free allocated memory on the device
	cudaFree(deviceResVec);
	cudaFree(deviceMaskVec);
	cudaFree(deviceMainVec);

	// Output timing to complete operation and container size
	std::cout << "GPU 1D Convolution computation time (container size: " << conSize << "):\n"
			  << std::chrono::duration_cast<std::chrono::microseconds>(opEnd - opStart).count() << " us\n"
			  << std::chrono::duration_cast<std::chrono::milliseconds>(opEnd - opStart).count() << " ms\n\n"
			  << "Returning to selection screen.\n\n"

			  << "#########################################################################\n" <<
			     "#########################################################################\n" <<
			     "#########################################################################\n\n";
}