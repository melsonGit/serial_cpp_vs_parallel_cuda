#pragma once
#ifndef EVENT_ARCHIVE
#define EVENT_ARCHIVE

#include <unordered_map>
#include <string>

// Ignore Error C26812: Enum Class values aren't implicitly cast to int, which is what we want
enum OperationEvents
{
	populateContainer,
	populateContainerComplete,
	startOperation,
	endOperation,
	validateResults,
	resultsValidated,
	recordResults,
	resultsRecorded,
};

inline const std::unordered_map<OperationEvents, std::string> eventArchive
{
	{populateContainer, "Populating containers."},
	{populateContainerComplete, "Containers populated."},
	{startOperation, "Starting operation."},
	{endOperation, "Operation complete."},
	{validateResults, "Starting result validation."},
	{resultsValidated, "Result validation complete."},
	{recordResults, "Starting output to file."},
	{resultsRecorded, "Output to file complete."},
};
#endif