#include "../../inc/matMulti/matMultiCore.h"

void matMultiCore()
{
	// Assign variable conSize with a user selected value
	int conSize { matMultiConSet(conSize) };
	size_t bytes{ conSize * conSize * sizeof(int) };

	// Assign native host input vectors (hostA & hostB) and the native host output vector (hostC) a container size of conSize * conSize
	std::vector<int> hostA(conSize * conSize), hostB(conSize * conSize), hostC(conSize * conSize);
	
	// Populate vectors
	std::cout << "\nMatrix Multiplication: Populating 1 of 2 host input vectors.\n";
	matMultiNumGen(hostA);
	std::cout << "\nMatrix Multiplication: Populating 2 of 2 host input vectors.\n";
	matMultiNumGen(hostB);

	std::cout << "\nMatrix Multiplication: Populating complete.\n";

	// Assign input device pointers (deviceA & deviceB) and output device pointer (deviceC) memory size of bytes
	int* deviceA, * deviceB, * deviceC;
	cudaMalloc(&deviceA, bytes);
	cudaMalloc(&deviceB, bytes);
	cudaMalloc(&deviceC, bytes);

	// Copy host input vector data to the device input pointers
	std::cout << "\nMatrix Multiplication: Copying data from host to device.\n";

	cudaMemcpy(deviceA, hostA.data(), bytes, cudaMemcpyHostToDevice);
	cudaMemcpy(deviceB, hostB.data(), bytes, cudaMemcpyHostToDevice);

	// Initialise threads per CTA (Compute Thread Array) dimension
	int THREADS { 32 };

	// Initialise blocks per grid dimension for threads to operate in
	int BLOCKS {conSize / THREADS};

	// Use dim3 structs for block and grid dimensions
	dim3 threads(THREADS, THREADS);
	dim3 blocks(BLOCKS, BLOCKS);

	// Start the clock
	clock_t opStart { clock() };

	// Launch kernel
	std::cout << "\nMatrix Multiplication: Starting operation.\n";

	matMultiFunc<<<blocks, threads>>>(deviceA, deviceB, deviceC, conSize);
	cudaDeviceSynchronize();

	std::cout << "\nMatrix Multiplication: Operation complete.\n";

	clock_t opEnd{ clock() };

	std::cout << "\nMatrix Multiplication: Copying results from device to host.\n";

	// Copy back to the host
	cudaMemcpy(hostC.data(), deviceC, bytes, cudaMemcpyDeviceToHost);

	std::cout << "\nMatrix Multiplication: Copying complete.\n";

	// Verify result on CPU
	matMultiCheck(hostA, hostB, hostC, conSize);

	std::cout << "\nMatrix Multiplication: Freeing device memory.\n\n";

	// Free memory on device
	cudaFree(deviceA);
	cudaFree(deviceB);
	cudaFree(deviceC);

	// Calculate overall time spent to complete operation
	double completionTime{ (opEnd - opStart) / (double)CLOCKS_PER_SEC };

	// Output timing to complete operation and container size
	std::cout << completionTime << "s Matrix Multiplication computation time, with a container size of " << conSize * conSize << ".\n\n";
	std::cout << "Returning to selection screen.\n\n";

	std::cout << "#########################################################################\n" <<
				 "#########################################################################\n" <<
			     "#########################################################################\n\n";
}