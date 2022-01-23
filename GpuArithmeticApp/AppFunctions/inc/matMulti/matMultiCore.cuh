#pragma once
#ifndef MAT_MULTI_CORE
#define MAT_MULTI_CORE

#include "matMultiCheck.h"
#include "matMultiConSet.h"
#include "matMultiFunc.cuh"
#include "matMultiNumGen.h"

#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <iostream>
#include <vector>

void matMultiCore();

#endif
