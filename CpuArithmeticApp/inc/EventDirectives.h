#pragma once
#ifndef EVENT_DIRECTIVES
#define EVENT_DIRECTIVES

#include <unordered_map>
#include <string>

// Ignore Error C26812: Enum Class values aren't implicitly cast to int, which is what we want
enum EventDirectives
{
	populateContainer = 0,
	populateContainerComplete,
	startOperation,
	endOperation,
	validateResults,
	resultsValidated,
};

inline const std::unordered_map<EventDirectives, std::string> eventDirectiveMap
{
	{populateContainer, "Populating containers."},
	{populateContainerComplete, "Containers populated."},
	{startOperation, "Starting operation."},
	{endOperation, "Operation complete."},
	{validateResults, "Starting result validation."},
	{resultsValidated, "Result validation complete."},
};
#endif