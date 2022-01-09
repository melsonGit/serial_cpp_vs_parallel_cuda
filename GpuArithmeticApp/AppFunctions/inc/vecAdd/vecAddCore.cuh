#pragma once
#ifndef VEC_ADD_CORE
#define VEC_ADD_CORE

#include <iostream>
#include <vector>

#include "vecAddConSet.h"
#include "vecAddNumGen.h"
#include "vecAddFunc.cuh"
#include "vecAddCheck.h"
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

void vecAddCore();

#endif
