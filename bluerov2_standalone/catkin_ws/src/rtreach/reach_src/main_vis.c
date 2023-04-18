// example call: ./uuv 10 -1960.9 42.01 0.175 11.21 0.335
// example call output:
// started!
// Argc: 7
// runtime: 10 ms
// x_0[0]: -1960.900000
// x_0[1]: 42.010000
// x_0[2]: 0.175000
// u_0[0]: 11.210000
// u_0[1]: 0.335000

#include <stdio.h>
#include <stdlib.h>
#include <stdbool.h>
#include "util.h"
#include "main.h"
#include "uuv_model.h"
#include "uuv_safety.h"
#include "simulate_uuv_plots.h"

const int state_n = 3; // state dimension

int main( int argc, const char* argv[] )
{
	DEBUG_PRINT("started!\n\r");

	int runtimeMs = 0;
	REAL startState[3] = {0.0, 0.0, 0.0};
    REAL control_input[2] = {0.0,0.0};

	DEBUG_PRINT("Argc: %d\n\r", argc);

	if (argc < 7) {
		printf("Error: not enough input arguments!\n\r");
		return 0;
	}
	else {
		runtimeMs = atoi(argv[1]);
		startState[0] = atof(argv[2]);
		startState[1] = atof(argv[3]);
		startState[2] = atof(argv[4]);
        control_input[0] = atof(argv[5]);
        control_input[1] = atof(argv[6]);
		DEBUG_PRINT("runtime: %d ms\n\rx_0[0]: %f\n\rx_0[1]: %f\n\rx_0[2]: %f\n\ru_0[0]: %f\n\ru_0[1]: %f\n\r\n", runtimeMs, startState[0], startState[1], startState[2],control_input[0],control_input[1]);
	}

    
	REAL uh = control_input[0];
    REAL uv = control_input[1];
   
    // simTime 
    REAL timeToSafe = 2.0;
    // startMs 
    int startMs = 0;

    // run reachability analysis test 
	HyperRectangle reach_hull = runReachability_uuv_vis(startState, timeToSafe, runtimeMs, startMs,uh,uv);
    println(&reach_hull);

	printf("\n");
    printf("num VisStates: %d\n",num_intermediate);
    printf("total encountered intermediate: %d\n",total_intermediate);
	for (int i =0; i < num_intermediate; i+=2)
	{
		println(&VisStates[i]);
    }

    getSimulatedSafeTime(startState,uh,uv);
    printf("\n");
    // // print the hull 
	// println(&reach_hull);
	// deallocate_2darr(file_rows,file_columns);
	// deallocate_obstacles(obstacle_count);


	return 0;
}