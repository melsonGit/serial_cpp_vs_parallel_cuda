#pragma once
#ifndef OPERATION_TIMER
#define OPERATION_TIMER

#include <chrono>

class OperationTimer
{

	using Clock = std::chrono::steady_clock;

private:

	// Creating OperationTimer object will start the clock
	std::chrono::time_point<Clock> mStartTimer{ Clock::now() };

	unsigned long long mElapsedTimeUs{};
	unsigned long long mElapsedTimeMs{};

	void elapsedMicroseconds();
	void elapsedMilliseconds();

public:

	void resetStartTimer();
	void collateElapsedTimes();
};
#endif