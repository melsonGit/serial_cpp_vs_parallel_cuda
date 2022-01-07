#include "../../inc/oneConv/oneConvCore.cuh"

void oneConvCore()
{
	// Assign variable conSize with a user selected value
	int conSize{ oneConvConSet(conSize) };
	// Size of the vector in bytes
	int bytes_n = conSize * sizeof(int);
	// Size of mask in bytes
	int bytes_m { MASK_ONE_DIM * sizeof(int) };

	// Allocate main vector and resultant vector with size conSize
	std::vector<int> hostMainVec(conSize), hostResVec(conSize);
	// Allocate mask vector with MASK_ONE_DIM
	std::vector<int> hostMaskVec(MASK_ONE_DIM);

	// Popluate main vector and mask vector
	std::cout << "\n1D Convolution: Populating main vector.\n";
	oneConvNumGen(hostMainVec);
	std::cout << "\n1D Convolution: Populating mask vector.\n";
	oneConvNumGen(hostMaskVec);

	std::cout << "\n1D Convolution: Populating complete.\n";

	// Allocate space on the device
	int* deviceMainVec, * deviceMaskVec, * deviceResVec;
	cudaMalloc(&deviceMainVec, bytes_n);
	cudaMalloc(&deviceMaskVec, bytes_m);
	cudaMalloc(&deviceResVec, bytes_n);

	// Copy host input vector data to the device input pointers
	std::cout << "\n1D Convolution: Copying data from host to device.\n";

	// Copy the data to the device
	cudaMemcpy(deviceMainVec, hostMainVec.data(), bytes_n, cudaMemcpyHostToDevice);
	cudaMemcpy(deviceMaskVec, hostMaskVec.data(), bytes_m, cudaMemcpyHostToDevice);

	// Threads per TB (thread blocks)
	int THREADS = 256;

	// Number of TBs with padding
	int BLOCKS{ (conSize + THREADS - 1) / THREADS };

	// Start clock
	clock_t opStart{ clock() };

	// Launch kernel
	std::cout << "\n1D Convolution: Starting operation.\n";

	// Call the kernel
	oneConvFunc << <BLOCKS, THREADS >> > (deviceMainVec, deviceMaskVec, deviceResVec, conSize);

	std::cout << "\n1D Convolution: Operation complete.\n";

	// Stop clock
	clock_t opEnd{ clock() };

	std::cout << "\n1D Convolution: Copying results from device to host.\n";

	// Copy back to the host
	cudaMemcpy(hostResVec.data(), deviceResVec, bytes_n, cudaMemcpyDeviceToHost);

	oneConvCheck(hostMainVec.data(), hostMaskVec.data(), hostResVec.data(), conSize);

	std::cout << "\n1D Convolution: Freeing device memory.\n\n";

	// Free allocated memory on the device and host
	cudaFree(deviceResVec);
	cudaFree(deviceMaskVec);
	cudaFree(deviceMainVec);

	// Calculate overall time spent to complete operation
	double completionTime{ (opEnd - opStart) / (double)CLOCKS_PER_SEC };

	// Output timing to complete operation and container size
	std::cout << completionTime << "s 1D Convolution computation time, with a container size of " << conSize << ".\n\n";
	std::cout << "Returning to selection screen.\n\n";

	std::cout << "#########################################################################\n" <<
				 "#########################################################################\n" <<
				 "#########################################################################\n\n";
}