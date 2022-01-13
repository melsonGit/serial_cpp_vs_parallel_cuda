#pragma once
#ifndef ONE_CONV_CORE
#define ONE_CONV_CORE

#include <iostream>
#include <vector>
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include "oneConvConSet.h"
#include "oneConvNumGen.h"
#include "oneConvFunc.cuh"
#include "oneConvCheck.h"

void oneConvCore();

#endif
