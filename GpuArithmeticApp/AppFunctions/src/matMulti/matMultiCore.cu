#include "../../inc/matMulti/matMultiCore.cuh"

void matMultiCore()
{
	// Assign variable conSize with a user selected value
	int conSize { matMultiConSet(conSize) };

	// Initialise memory allocation variable
	size_t vecMem { conSize * conSize * sizeof(int) };

	// Assign native host input vectors (hostA & hostB) and the native host output vector (hostC) a container size of conSize * conSize
	std::vector<int> hostVecA(conSize * conSize), hostVecB(conSize * conSize), hostResultVec(conSize * conSize);
	
	// Populate vectors
	std::cout << "\nMatrix Multiplication: Populating 1 of 2 host input vectors.\n";
	matMultiNumGen(hostVecA);
	std::cout << "\nMatrix Multiplication: Populating 2 of 2 host input vectors.\n";
	matMultiNumGen(hostVecB);

	std::cout << "\nMatrix Multiplication: Populating complete.\n";

	// Assign input device pointers (deviceA & deviceB) and output device pointer (deviceC) memory size of vecMem
	int* deviceVecA, * deviceVecB, * deviceResultVec;
	cudaMalloc(&deviceVecA, vecMem);
	cudaMalloc(&deviceVecB, vecMem);
	cudaMalloc(&deviceResultVec, vecMem);

	// Copy host input vector data to the device input pointers
	std::cout << "\nMatrix Multiplication: Copying data from host to device.\n";

	cudaMemcpy(deviceVecA, hostVecA.data(), vecMem, cudaMemcpyHostToDevice);
	cudaMemcpy(deviceVecB, hostVecB.data(), vecMem, cudaMemcpyHostToDevice);

	// Initialise threads per CTA (Compute Thread Array) dimension
	int THREADS { 32 };

	// Initialise blocks per grid dimension for threads to operate in
	int BLOCKS { (conSize + THREADS - 1) / THREADS };

	// Use dim3 structs for block and grid dimensions
	dim3 threads(THREADS, THREADS);
	dim3 blocks(BLOCKS, BLOCKS);

	// Start the clock
	clock_t opStart { clock() };

	// Launch kernel
	std::cout << "\nMatrix Multiplication: Starting operation.\n";

	matMultiFunc<<<blocks, threads>>>(deviceVecA, deviceVecB, deviceResultVec, conSize);

	std::cout << "\nMatrix Multiplication: Operation complete.\n";
	std::cout << "\nMatrix Multiplication: Copying results from device to host.\n";

	// Copy back to the host
	cudaMemcpy(hostResultVec.data(), deviceResultVec, vecMem, cudaMemcpyDeviceToHost);

	std::cout << "\nMatrix Multiplication: Copying complete.\n";

	clock_t opEnd{ clock() };

	// Verify result on CPU
	matMultiCheck(hostVecA, hostVecB, hostResultVec, conSize);

	std::cout << "\nMatrix Multiplication: Freeing device memory.\n\n";

	// Free memory on device
	cudaFree(deviceVecA);
	cudaFree(deviceVecB);
	cudaFree(deviceResultVec);

	// Calculate overall time spent to complete operation
	double completionTime{ ((static_cast<double>(opEnd)) - (static_cast<double>(opStart))) / (double)CLOCKS_PER_SEC };

	// Output timing to complete operation and container size
	std::cout << completionTime << "s Matrix Multiplication computation time, with a container size of " << conSize * conSize << ".\n\n";
	std::cout << "Returning to selection screen.\n\n";

	std::cout << "#########################################################################\n" <<
				 "#########################################################################\n" <<
			     "#########################################################################\n\n";
}