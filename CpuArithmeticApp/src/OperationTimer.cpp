#include "../inc/OperationTimer.h"

void OperationTimer::resetStartTimer()
{
	mStartTimer = Clock::now();

	// Reset elapsed vars to 0
	mElapsedTimeUs = 0;
	mElapsedTimeMs = 0;
}

void OperationTimer::elapsedMicroseconds() // measured in us
{
	this->mElapsedTimeUs = std::chrono::microseconds((Clock::now() - mStartTimer).count()).count();
}

void OperationTimer::elapsedMilliseconds() // measured in ms
{
	this->mElapsedTimeMs = std::chrono::milliseconds((Clock::now() - mStartTimer).count()).count();
}

void OperationTimer::collateElapsedTimes()
{
	this->elapsedMicroseconds();
	this->elapsedMilliseconds();
}