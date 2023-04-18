// Patrick Musau
// Vehicle Bicycle model header

#ifndef BICYCLE_PARAM_DISTURBANCE_H_
#define BICYCLE_PARAM_DISTURBANCE_H_

#include "main.h"
#include "geometry.h"
#include <stdbool.h>



// run reachability for a given wall timer (or iterations if negative)
bool runReachability_bicycle_disturbance(REAL* start, REAL simTime, REAL wallTimeMs, REAL startMs,REAL heading, REAL throttle,HyperRectangle VisStates[],int  *total_intermediate,int max_intermediate,bool plot, REAL disturbance);
bool check_safety_disturbance(HyperRectangle* rect, REAL (*cone)[2]);
#endif