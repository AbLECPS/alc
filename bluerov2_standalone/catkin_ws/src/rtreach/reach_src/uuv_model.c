#include "uuv_model.h"
#include "main.h"
#include "face_lift.h"
#include "util.h"
#include "simulate_uuv.h"
#include "uuv_safety.h"
#include <stdio.h>
#include <string.h> 
#include <stdlib.h>


// a note from the f1tenth simulator 
// the car is 0.5 m long in the x direction 
// 0.3 long in the y direction




// do face lifting with the given settings, iteratively improving the computation
// returns true if the reachable set of states is satisfactory according to the
// function you provide in LiftingSettings (reachedAtIntermediateTime, reachedAtFinalTime)

bool face_lifting_iterative_improvement_uuv(int startMs, LiftingSettings* settings, REAL heading, REAL speed);

// helper function to check safety
// bool check_safety(HyperRectangle* rect, double (*box)[2]); 


// function that stops simulation after two seconds
bool shouldStop(REAL state[NUM_DIMS], REAL simTime, void* p)
{
	bool rv = false;
    REAL maxTime = 2.0f;
    // stop if the maximum simulation time 
	if (simTime >= maxTime)
	{
		rv = true;

		REAL* stopTime = (REAL*)p;
		*stopTime = -1;
	}

	return rv;
}



// Simulation 
REAL getSimulatedSafeTime(REAL start[3],REAL heading,REAL speed)
{
	REAL stepSize = 0.02f;
	REAL rv = 0.0f;

	simulate_uuv(start, heading,speed,stepSize, shouldStop, (void*)&rv); // TODO: look here

	//DEBUG_PRINT("time until simulation reaches safe state = %f\n", rv);

	return rv;
}

// called on states reached during the computation
bool intermediateState(HyperRectangle* r)
{
	bool allowed = true;
	//const REAL FIFTEEN_DEGREES_IN_RADIANS = 0.2618;

	// bloat the box for the width of the car
	r->dims[0].min = r->dims[0].min  - 0.99;
	r->dims[0].max = r->dims[0].max  + 0.99;
	r->dims[1].min = r->dims[1].min  - 0.115;
	r->dims[1].max = r->dims[1].max  + 0.115;

	
	allowed = check_safety_obstacles(r);

	// reset it
	r->dims[0].min = r->dims[0].min  + 0.99;
	r->dims[0].max = r->dims[0].max  - 0.99;
	r->dims[1].min = r->dims[1].min  + 0.115;
	r->dims[1].max = r->dims[1].max  - 0.115;

	//if(!allowed)
	//	printf("unsafe....\n");
	return allowed;
}

// This function enumerates all of the corners of the current HyperRectangle and 
// returns whether or not any of the points lies outside of the ellipsoid
bool finalState(HyperRectangle* rect)
{

	return intermediateState(rect);
}




bool runReachability_uuv(REAL* start, REAL simTime, REAL wallTimeMs, REAL startMs,REAL heading, REAL speed)
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

	
	

	set.initialStepSize = iss; 
	set.maxRectWidthBeforeError = 100;

	set.reachedAtFinalTime = finalState;
	set.reachedAtIntermediateTime = intermediateState;
	set.restartedComputation = 0; //restartedComputation;

	return face_lifting_iterative_improvement_uuv(startMs, &set,heading, speed);
}



