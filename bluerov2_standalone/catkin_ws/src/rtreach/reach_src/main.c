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

const int state_n = 3; // state dimension

// This particular example also needs to know where the walls are
// const char * filepath= "../ros_src/rtreach/obstacles/porto_obstacles.txt";

int main( int argc, const char* argv[] )
{
	DEBUG_PRINT("started!\n\r");

	int runtimeMs = 0;
	REAL startState[3] = {0.0, 0.0, 0.0};
    REAL control_input[2] = {0.0,0.0};

	DEBUG_PRINT("Argc: %d\n\r", argc);

	if (argc < 5) {
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
		DEBUG_PRINT("runtime: %d ms\n\rx_0[0]: %f\n\rx_0[1]: %f\n\rx_0[2]: %f\n\ru_0[0]: %f\n\ru_0[1]: %f\n\r\n", runtimeMs, startState[0], startState[1], startState[2], control_input[0],control_input[1]);
	}

    
    REAL uh = control_input[0];
    REAL uv = control_input[1];
    // simulate the car with a constant input passed from the command line
    getSimulatedSafeTime(startState,uh,uv);
    printf("\n");

	// // location of obstacles in our scenario
	// int num_obstacles = 5;
	// double points[5][2] = {{2.0,2.0},{4.7,2.7},{11.36,-1.46},{3.0,6.4},{-9.64,2.96}};
	// allocate_obstacles(num_obstacles,points);
    // simTime 
    REAL timeToSafe = 2.0;

    // startMs 
    int startMs = 0;

	// load the wall points into the global variable
	//load_wallpoints(filepath,true);
    
    // run reachability analysis test 
	bool safe = runReachability_uuv(startState, timeToSafe, runtimeMs, startMs,uh,uv);
	// //int runtimeMs = 20; // run for 20 milliseconds

	DEBUG_PRINT("done, result = %s\n", safe ? "safe" : "unsafe");
	// deallocate_2darr(file_rows,file_columns);
	// deallocate_obstacles(obstacle_count);

	return 0;
}