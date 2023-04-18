#include <stdio.h>
#include <float.h>
#include <math.h>
#include <stdlib.h>
#include "bicycle_model_disturbance.h"

#include "main.h"
#include "face_lift_obstacle.h"
#include "util.h"

#include <stdio.h>
#include <float.h>
#include <math.h>
#include <stdlib.h>
#include "bicycle_model_degraded.h"

#include "main.h"
#include "face_lift_obstacle.h"
#include "util.h"

// do face lifting with the given settings, iteratively improving the computation
// returns true if the reachable set of states is satisfactory according to the
// function you provide in LiftingSettings (reachedAtIntermediateTime, reachedAtFinalTime)
bool face_lifting_iterative_improvement_bicycle_disturbance(int startMs, LiftingSettings* settings, REAL heading_input, REAL throttle,HyperRectangle VisStates[],int  *total_intermediate,int max_intermediate,bool plot,REAL disurbance);

void restartedComputation_disturbance(int  *total_intermediate)
{
	// reset the counter of intermediate states 
	*total_intermediate = 0;
}


// called on states reached during the computation
bool intermediateState_disturbance(HyperRectangle* r,HyperRectangle VisStates[],int  *total_intermediate,int max_intermediate)
{
	if(*total_intermediate < max_intermediate)
	{
		VisStates[*total_intermediate] = *r;
		*total_intermediate+=1;
	}
	return true;
}


bool finalState_disturbance(HyperRectangle* rect, HyperRectangle VisStates[],int  *total_intermediate,int max_intermediate)
{
	if(*total_intermediate < max_intermediate)
	{
		VisStates[*total_intermediate] = *rect;
	}
    *total_intermediate=*total_intermediate+1;
	return true;
}


bool runReachability_bicycle_disturbance(REAL* start, REAL simTime, REAL wallTimeMs, REAL startMs,REAL heading, REAL throttle,HyperRectangle VisStates[],int  *total_intermediate,int max_intermediate,bool plot, REAL disturbance)
{
    LiftingSettings set;
	for (int d = 0; d < NUM_DIMS; ++d)
	{
		set.init.dims[d].min = start[d];
		set.init.dims[d].max = start[d];
	}

	set.reachTime = simTime;
	set.maxRuntimeMilliseconds = wallTimeMs;

	REAL iss = set.reachTime;
	iss = iss * 0.10f;
	if(iss>0.3)
	{
		iss = 0.3;
	}

	
	

	set.initialStepSize = iss; 
	set.maxRectWidthBeforeError = 100;

	set.reachedAtFinalTime = finalState_disturbance;
	set.reachedAtIntermediateTime = intermediateState_disturbance;
	set.restartedComputation = restartedComputation_disturbance; 

    return face_lifting_iterative_improvement_bicycle_disturbance(startMs, &set,heading, throttle,VisStates, total_intermediate,max_intermediate,plot,disturbance);
}

bool check_safety_disturbance(HyperRectangle* rect, REAL (*cone)[2])
{
	
	REAL l1[2] = {rect->dims[0].min- 0.2285,rect->dims[1].max+ 0.169};
    REAL r1[2] = {rect->dims[0].max+ 0.2285,rect->dims[1].min- 0.169};

	REAL l2[2] = {cone[0][0],cone[1][1]};
    REAL r2[2] = {cone[0][1],cone[1][0]};

	if (l1[0] >= r2[0] || l2[0] >= r1[0]) 
        return true; 
    
    if (l1[1] <= r2[1] || l2[1] <= r1[1]) 
        return true; 

	return false;
}