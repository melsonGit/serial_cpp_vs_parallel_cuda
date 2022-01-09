#include "../../inc/vecAdd/vecAddCore.cuh"

void vecAddCore()
{
	// Assign variable conSize with a user selected value
	int conSize{ vecAddConSet(conSize) };
	size_t vecMem { sizeof(int) * conSize };

	// Assign input vectors (a & b) and the output vector (c) a container size of conSize
	std::vector<int> hostVecA(conSize), hostVecB(conSize), hostResultVec(conSize);

	// Populate vectors
	std::cout << "\nVector Addition: Populating 1 of 2 input vectors.\n";
	vecAddNumGen(hostVecA);
	std::cout << "\nVector Addition: Populating 2 of 2 input vectors.\n";
	vecAddNumGen(hostVecB);

	std::cout << "\nVector Addition: Populating complete.\n";

	// Allocate memory on the device (GPU)
	int* deviceVecA, * deviceVecB, * deviceResultVec;
	cudaMalloc(&deviceVecA, vecMem);
	cudaMalloc(&deviceVecB, vecMem);
	cudaMalloc(&deviceResultVec, vecMem);

	std::cout << "\nVector Addition: Copying data from host to device.\n";

	// Copy data from the host to the device
	cudaMemcpy(deviceVecA, hostVecA.data(), vecMem, cudaMemcpyHostToDevice);
	cudaMemcpy(deviceVecB, hostVecB.data(), vecMem, cudaMemcpyHostToDevice);

	// Threads per Cooperative Thread Array (CTA; 1024)
	int THREADS = 1024;

	// CTAs per Grid
	int BLOCKS = (conSize + THREADS - 1) / THREADS;

	// Start clock
	clock_t opStart{ clock() };

	std::cout << "\nVector Addition: Starting operation.\n";

	// Launch the kernel on the GPU
	vecAddFunc << <BLOCKS, THREADS >> > (deviceVecA, deviceVecB, deviceResultVec, conSize);

	std::cout << "\nVector Addition: Operation complete.\n";

	// Stop clock
	clock_t opEnd{ clock() };

	std::cout << "\nVector Addition: Copying results from device to host.\n";

	// Copy sum vector from device to host
	cudaMemcpy(hostResultVec.data(), deviceResultVec, vecMem, cudaMemcpyDeviceToHost);

	// Check output vector contents
	vecAddCheck(hostVecA, hostVecB, hostResultVec, conSize);

	std::cout << "\nVector Addition: Freeing device memory.\n\n";

	// Free memory on device
	cudaFree(deviceVecA);
	cudaFree(deviceVecB);
	cudaFree(deviceResultVec);

	// Calculate overall time spent to complete operation
	double completionTime{ (opEnd - opStart) / (double)CLOCKS_PER_SEC };

	// Output timing to complete operation and container size
	std::cout << completionTime << "s Vector Addition computation time, with a container size of " << conSize << ".\n\n";
	std::cout << "Returning to selection screen.\n\n";

	std::cout << "#########################################################################\n" <<
			     "#########################################################################\n" <<
				 "#########################################################################\n\n";
}