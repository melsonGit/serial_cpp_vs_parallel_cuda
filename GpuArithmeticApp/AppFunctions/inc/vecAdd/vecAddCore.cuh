#pragma once
#ifndef VEC_ADD_CORE
#define VEC_ADD_CORE

#include "vecAddCheck.h"
#include "vecAddConSet.h"
#include "vecAddFunc.cuh"
#include "vecAddNumGen.h"

#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <iostream>
#include <vector>

void vecAddCore();

#endif