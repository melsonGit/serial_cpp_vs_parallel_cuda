#include "../inc/OperationTimeHandler.h"

using precisionType = unsigned long long;

void OperationTimeHandler::resetStartTimer()
{
	this->mStartTimer = operationClock::now();

	// Reset elapsed vars to 0
	this->mElapsedTimeUs = 0;
	this->mElapsedTimeMs = 0;
	this->mElapsedTimeS = 0;
}
void OperationTimeHandler::elapsedMicroseconds() // measured in us
{
	this->mElapsedTimeUs = std::chrono::duration_cast<std::chrono::microseconds>(operationClock::now() - this->mStartTimer).count();
}
void OperationTimeHandler::elapsedMilliseconds() // measured in ms
{
	this->mElapsedTimeMs = std::chrono::duration_cast<std::chrono::milliseconds>(operationClock::now() - this->mStartTimer).count();
}
void OperationTimeHandler::elapsedSeconds()
{
	this->mElapsedTimeS = std::chrono::duration_cast<std::chrono::seconds>(operationClock::now() - this->mStartTimer).count();
}
void OperationTimeHandler::collectElapsedTimeData()
{
	this->elapsedMicroseconds();
	this->elapsedMilliseconds();
	this->elapsedSeconds();
}

const precisionType& OperationTimeHandler::getElapsedMicroseconds() const
{
	return this->mElapsedTimeUs;
}
const precisionType& OperationTimeHandler::getElapsedMilliseconds() const
{
	return this->mElapsedTimeMs;
}
const precisionType& OperationTimeHandler::getElapsedSeconds() const
{
	return this->mElapsedTimeS;
}