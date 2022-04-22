#pragma once
#ifndef ONE_CONV_CORE
#define ONE_CONV_CORE

#include "../maskAttributes/maskAttributes.h"
#include "oneConvCheck.h"
#include "oneConvConSet.h"
#include "oneConvFunc.cuh"
#include "oneConvNumGen.h"

#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <chrono>
#include <iostream>
#include <vector>

void oneConvCore();

#endif