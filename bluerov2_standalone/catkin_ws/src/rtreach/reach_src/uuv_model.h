// Patrick Musau
// 11-2020
// UUV model header

#ifndef BICYCLE_H_
#define BICYCLE_H_

#include "main.h"
#include "geometry.h"
#include <stdbool.h>



// run reachability for a given wall timer (or iterations if negative)
bool runReachability_uuv(REAL* start, REAL simTime, REAL wallTimeMs, REAL startMs,REAL heading, REAL speed);
HyperRectangle runReachability_uuv_vis(REAL* start, REAL simTime, REAL wallTimeMs, REAL startMs,REAL heading, REAL speed);
REAL getSimulatedSafeTime(REAL start[3],REAL heading_input, REAL speed);

#endif