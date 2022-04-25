#pragma once
#ifndef OPERATION_TIMER
#define OPERATION_TIMER

#include <chrono>

class OperationTimer
{
	using operationClock = std::chrono::steady_clock;
	using precisionType = unsigned long long;

private:

	// Creating OperationTimer object will start the clock
	std::chrono::time_point<operationClock> mStartTimer{ operationClock::now() };

	precisionType mElapsedTimeUs{};
	precisionType mElapsedTimeMs{};
	precisionType mElapsedTimeS{};

	void elapsedMicroseconds();
	void elapsedMilliseconds();
	void elapsedSeconds();

public:

	void resetStartTimer();
	void collectElapsedTimeData();

	precisionType getElapsedMicroseconds() const;
	precisionType getElapsedMilliseconds() const;
	precisionType getElapsedSeconds() const;
};
#endif