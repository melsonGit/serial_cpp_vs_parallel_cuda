#include "../../inc/vecAdd/vecAddCore.cuh"

void vecAddCore()
{
	// Initialise and allocate variable conSize with a user selected value
	int conSize { vecAddConSet(conSize) };

	// Initialise and allocate input vectors (hostInputVecA & hostInputVecB) and output vector (hostResultVec) a container size of conSize
	std::vector<int> hostInputVecA(conSize), hostInputVecB(conSize), hostResultVec(conSize);

	// Populate input vectors
	std::cout << "\nVector Addition: Populating 1 of 2 input vectors.\n";
	vecAddNumGen(hostInputVecA);
	std::cout << "\nVector Addition: Populating 2 of 2 input vectors.\n";
	vecAddNumGen(hostInputVecB);

	std::cout << "\nVector Addition: Populating complete.\n";

	// Initialise bytesVecMem to used for allocating memory to device vars
	// This allows us to copy data host to device and vice versa.
	size_t bytesVecMem { sizeof(int) * conSize };

	// Allocate memory on the device using cudaMalloc
	int* deviceInputVecA, * deviceInputVecB, * deviceResultVec;
	cudaMalloc(&deviceInputVecA, bytesVecMem);
	cudaMalloc(&deviceInputVecB, bytesVecMem);
	cudaMalloc(&deviceResultVec, bytesVecMem);

	std::cout << "\nVector Addition: Copying data from host to device.\n";

	// Copy data from the host to the device using cudaMemcpy | .data() returns pointer to memory used by vector/array to store its owned elements
	cudaMemcpy(deviceInputVecA, hostInputVecA.data(), bytesVecMem, cudaMemcpyHostToDevice);
	cudaMemcpy(deviceInputVecB, hostInputVecB.data(), bytesVecMem, cudaMemcpyHostToDevice);

	// Threads per Cooperative Thread Array
	int THREADS { 32 };

	// No. CTAs per grid
	// Add padding | Enables compatibility with sample sizes not divisible by 32
	int BLOCKS { (conSize + THREADS - 1) / THREADS };

	// Start the clock
	clock_t opStart { clock() };

	std::cout << "\nVector Addition: Starting operation.\n";

	// Launch kernel on device
	vecAddFunc << <BLOCKS, THREADS >> > (deviceInputVecA, deviceInputVecB, deviceResultVec, conSize);

	std::cout << "\nVector Addition: Operation complete.\n";

	// Stop clock
	clock_t opEnd { clock() };

	std::cout << "\nVector Addition: Copying results from device to host.\n";

	// Copy data from device back to host using cudaMemcpy
	cudaMemcpy(hostResultVec.data(), deviceResultVec, bytesVecMem, cudaMemcpyDeviceToHost);

	std::cout << "\nVector Addition: Copying complete.\n";

	// Authenticate results on host
	vecAddCheck(hostInputVecA, hostInputVecB, hostResultVec, conSize);

	std::cout << "\nVector Addition: Freeing device memory.\n\n";

	// Free memory on device
	cudaFree(deviceInputVecA);
	cudaFree(deviceInputVecB);
	cudaFree(deviceResultVec);

	// Calculate overall time spent to complete operation
	double completionTime { ((static_cast<double>(opEnd)) - (static_cast<double>(opStart))) / (double)CLOCKS_PER_SEC };

	// Output timing to complete operation and container size
	std::cout << completionTime << "s Vector Addition computation time, with a container size of " << conSize << ".\n\n";
	std::cout << "Returning to selection screen.\n\n";

	std::cout << "#########################################################################\n" <<
			     "#########################################################################\n" <<
				 "#########################################################################\n\n";
}