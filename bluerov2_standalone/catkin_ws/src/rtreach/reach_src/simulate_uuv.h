// Patrick Musau
// 11-2020
// Simulate Header

#ifndef SIMULATE_H_
#define SIMULATE_H_

#include "dynamics_uuv_model.h"
#include "geometry.h"
#include "main.h"
#include <stdbool.h>

// simulate dynamics using Euler's method
void simulate_uuv(REAL point[NUM_DIMS], REAL heading, REAL speed,
			  REAL stepSize,
			  bool (*shouldStop)(REAL state[NUM_DIMS], REAL simTime, void* p),
			  void* param);

#endif
