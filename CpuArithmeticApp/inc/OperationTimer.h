#include <chrono>

class OperationTimer
{

	using Clock = std::chrono::steady_clock;

private:

	// Creating OperationTimer object will start the clock
	std::chrono::time_point<Clock> mStartTimer{ Clock::now() };

	unsigned long long mElapsedTimeUs{};
	unsigned long long mElapsedTimeMs{};

public:

	void resetStartTimer()
	{
		mStartTimer = Clock::now();

		// Reset elapsed vars to 0
		mElapsedTimeUs = 0;
		mElapsedTimeMs = 0;
	}

	void elapsedMicroseconds() // measured in us
	{
		this->mElapsedTimeUs = std::chrono::microseconds((Clock::now() - mStartTimer).count()).count();
	}

	void elapsedMilliseconds() // measured in ms
	{
		this->mElapsedTimeMs = std::chrono::milliseconds((Clock::now() - mStartTimer).count()).count();
	}
};