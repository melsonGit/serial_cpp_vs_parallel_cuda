#pragma once
#ifndef TWO_CONV_CORE
#define TWO_CONV_CORE

#include <iostream>
#include <vector>
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include "twoConvConSet.h"
#include "twoConvNumGen.h"
#include "twoConvFunc.cuh"
#include "twoConvCheck.h"
#include "../maskAttributes/maskAttributes.h"

void twoConvCore();

#endif