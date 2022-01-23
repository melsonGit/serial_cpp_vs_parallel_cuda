#pragma once
#ifndef TWO_CONV_CORE
#define TWO_CONV_CORE

#include "../maskAttributes/maskAttributes.h"
#include "twoConvCheck.h"
#include "twoConvConSet.h"
#include "twoConvFunc.cuh"
#include "twoConvNumGen.h"

#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <iostream>
#include <vector>

void twoConvCore();

#endif