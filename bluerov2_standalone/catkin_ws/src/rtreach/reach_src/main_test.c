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
#include "uuv_safety.h"

int main()
{
	allocate_obstacles(10);
    double obs1[2][2] = {{1.0,0.0},{0.0,1.0}};
    append_obstacle(obs1);
    print_obstacles();
    deallocate_obstacles();

	return 0;
}