#include <chrono>

using Clock = std::chrono::steady_clock;

class OperationTimer
{
private:


	// Start clock
	auto opStart{ Clock::now() };
	// Stop clock
	auto opEnd{ Clock::now() };

public:

	std::chrono::duration_cast<std::chrono::microseconds>(opEnd - opStart).count() << " us\n"
		<< std::chrono::duration_cast<std::chrono::milliseconds>(opEnd - opStart).count() <<
};