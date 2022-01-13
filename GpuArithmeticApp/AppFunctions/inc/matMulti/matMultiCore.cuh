#pragma once
#ifndef MAT_MULTI_CORE
#define MAT_MULTI_CORE

#include <iostream>
#include <vector>
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include "matMultiConSet.h"
#include "matMultiNumGen.h"
#include "matMultiFunc.cuh"
#include "matMultiCheck.h"

void matMultiCore();

#endif
