#include "../../inc/matMulti/matMultiCore.cuh"

void matMultiCore()
{
	// Initialise and allocate variable conSize with a user selected value
	int conSize { matMultiConSet(conSize) };

	// Initialise and allocate native host input vectors (hostInputVecA & hostInputVecB) and the native host output vector (hostResultVec) a container size of conSize * conSize
	std::vector<int> hostInputVecA(conSize * conSize), hostInputVecB(conSize * conSize), hostResultVec(conSize * conSize);
	
	// Populate input vectors
	std::cout << "\nMatrix Multiplication: Populating 1 of 2 host input vectors.\n";
	matMultiNumGen(hostInputVecA);
	std::cout << "\nMatrix Multiplication: Populating 2 of 2 host input vectors.\n";
	matMultiNumGen(hostInputVecB);

	std::cout << "\nMatrix Multiplication: Populating complete.\n";

	// Initialise bytesVecMem to used for allocating memory to device vars
	// This allows us to copy data host to device and vice versa.
	size_t bytesVecMem { conSize * conSize * sizeof(int) };

	// Allocate memory on the device using cudaMalloc
	int* deviceInputVecA, * deviceInputVecB, * deviceResultVec;
	cudaMalloc(&deviceInputVecA, bytesVecMem);
	cudaMalloc(&deviceInputVecB, bytesVecMem);
	cudaMalloc(&deviceResultVec, bytesVecMem);

	std::cout << "\nMatrix Multiplication: Copying data from host to device.\n";

	// Copy data from the host to the device using cudaMemcpy | .data() returns pointer to memory used by vector/array to store its owned elements
	cudaMemcpy(deviceInputVecA, hostInputVecA.data(), bytesVecMem, cudaMemcpyHostToDevice);
	cudaMemcpy(deviceInputVecB, hostInputVecB.data(), bytesVecMem, cudaMemcpyHostToDevice);

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

	std::cout << "\nMatrix Multiplication: Starting operation.\n";

	// Launch kernel on device
	matMultiFunc<<<blocks, threads>>>(deviceInputVecA, deviceInputVecB, deviceResultVec, conSize);

	std::cout << "\nMatrix Multiplication: Operation complete.\n";

	// Stop clock
	clock_t opEnd { clock() };

	std::cout << "\nMatrix Multiplication: Copying results from device to host.\n";

	// Copy data from device back to host using cudaMemcpy
	cudaMemcpy(hostResultVec.data(), deviceResultVec, bytesVecMem, cudaMemcpyDeviceToHost);

	std::cout << "\nMatrix Multiplication: Copying complete.\n";

	// Authenticate results on host
	matMultiCheck(hostInputVecA, hostInputVecB, hostResultVec, conSize);

	std::cout << "\nMatrix Multiplication: Freeing device memory.\n\n";

	// Free memory on device
	cudaFree(deviceInputVecA);
	cudaFree(deviceInputVecB);
	cudaFree(deviceResultVec);

	// Calculate overall time spent to complete operation
	double completionTime { ((static_cast<double>(opEnd)) - (static_cast<double>(opStart))) / (double)CLOCKS_PER_SEC };

	// Output timing to complete operation and container size
	std::cout << completionTime << "s Matrix Multiplication computation time, with a container size of " << conSize * conSize << ".\n\n";
	std::cout << "Returning to selection screen.\n\n";

	std::cout << "#########################################################################\n" <<
				 "#########################################################################\n" <<
			     "#########################################################################\n\n";
}