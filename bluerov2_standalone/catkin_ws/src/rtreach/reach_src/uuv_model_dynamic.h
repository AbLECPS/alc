#include "main.h"
#include "geometry.h"
#include <stdbool.h>



// run reachability for a given wall timer (or iterations if negative)
bool runReachability_uuv_dyn(REAL* start, REAL simTime, REAL wallTimeMs, REAL startMs,REAL heading, REAL speed,HyperRectangle VisStates[],int  *total_intermediate,int max_intermediate,bool plot);
REAL getSimulatedSafeTime(REAL start[3],REAL heading_input, REAL speed);
bool check_safety(HyperRectangle* rect, REAL (*cone)[2]);