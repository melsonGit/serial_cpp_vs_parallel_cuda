#include "../inc/OperationTimer.h"

using precisionType = unsigned long long;

void OperationTimer::resetStartTimer()
{
	this->mStartTimer = operationClock::now();

	// Reset elapsed vars to 0
	this->mElapsedTimeUs = 0;
	this->mElapsedTimeMs = 0;
	this->mElapsedTimeS = 0;
}
void OperationTimer::elapsedMicroseconds() // measured in us
{
	this->mElapsedTimeUs = std::chrono::duration_cast<std::chrono::microseconds>(operationClock::now() - this->mStartTimer).count();
}
void OperationTimer::elapsedMilliseconds() // measured in ms
{
	this->mElapsedTimeMs = std::chrono::duration_cast<std::chrono::milliseconds>(operationClock::now() - this->mStartTimer).count();
}
void OperationTimer::elapsedSeconds()
{
	this->mElapsedTimeS = std::chrono::duration_cast<std::chrono::seconds>(operationClock::now() - this->mStartTimer).count();
}
void OperationTimer::collectElapsedTimeData()
{
	this->elapsedMicroseconds();
	this->elapsedMilliseconds();
	this->elapsedSeconds();
}

precisionType OperationTimer::getElapsedMicroseconds() const
{
	return this->mElapsedTimeUs;
}
precisionType OperationTimer::getElapsedMilliseconds() const
{
	return this->mElapsedTimeMs;
}
precisionType OperationTimer::getElapsedSeconds() const
{
	return this->mElapsedTimeS;
}

