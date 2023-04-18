#include "dynamics_bicycle.h"
#include "dynamics_bicycle_model_disturbance.h"
#include "util.h"
#include "interval.h"

#include <stdlib.h>
#include <stdbool.h>
#include <stdio.h>
#include <math.h>

#ifdef DYNAMICS_BICYCLE_MODEL

// a bicycle model to model the car's dynamics. The bicycle model is a standard model for cars with front steering. 
// This model tracks well for slow speeds

// x' = v * cos(theta + beta)
// y' = v * sin(theta + beta)
// v' = -ca * v + ca*cm*(u - ch)
// theta' = v * (cos(beta)/(lf+lr)) * tan(delta)
// beta = arctan(lr*tan(delta)/(lf+lr))

// for simplicity we are going to assume beta is 0. 

// Inputs, there are two inputs
// u is the throttle input
// delta is the heading input

// v is linear velocity 
// theta car's orientation 
// Beta is the car's slip angle 
// x and y are car's position
// u is throttle input 
// delta is the heading input 
// ca is an acceleration 
// cm is car motor constant
// ch is hysterisis constant
// lf, lr are the distances from the car's center of mass 

// parameters from (https://repository.upenn.edu/cgi/viewcontent.cgi?article=1908&context=cis_papers)
// ca = 1.633
// cm = 0.2
// ch = 4
// lf = 0.225
// lr = 0.225
// u = 16 constant speed (2.4 m/s)

// state vector x,y,v,theta

// double params[3] = {0.0677,7.2439, 0.7941};
// double params[3] = {86.4728, 77.1331, 0.9629};
// double params[3] = {96.4667, 77.6160, 0.9642};
// double params[3] = {0.0006, 100.7298,0.9286};
double params_disturbance[3] = {0.0916,5.9942,0.7788};
// double params[3] = {0.0450, 27.7782, 1.0505};


// implement the derivative using interval arithmetic
double get_derivative_bounds_bicycle_disturbance(HyperRectangle* rect, int faceIndex,REAL heading_input, REAL throttle, REAL disturbance)
{   
    // parameter uncertainty as a percentage
    REAL parmater_uncertainty = disturbance;
    REAL param_up = (1.0+ parmater_uncertainty);
    REAL param_down = (1.0 - parmater_uncertainty);
    disturbance = 0.0;
    REAL u = throttle;
    REAL delta = heading_input;
    REAL ca = params_disturbance[0];                   
    REAL cm = params_disturbance[1];        
    REAL ch = params_disturbance[2];  // These params come from sysid
    REAL lf = 0.2285;
    REAL lr = 0.169;

    int dim = faceIndex / 2;
	bool isMin = (faceIndex % 2) == 0;

    // Interval rv.min = rv.max =0
    Interval rv = new_interval_v(0);
    Interval v = rect->dims[2];
    Interval theta = rect->dims[3];
    Interval disturbance_interval = new_interval(-disturbance/2.0,disturbance/2.0);

    // beta = arctan(lr*tan(delta)/(lf+lr))

    if( dim == 0 ) 
    {
        // x' = v * cos(theta + beta)
        Interval A = mul_interval(v,cos_interval(theta));
        rv = add_interval(disturbance_interval,A);

    }
    else if(dim == 1)
    {
        // y' = v * sin(theta + beta)
        rv = mul_interval(v,sin_interval(theta));
    }
    else if(dim ==2)
    {
        // v' = -ca * v + ca*cm*(u - ch)

        Interval A = mul_interval(v, new_interval(-(ca*param_up),-(ca*param_down)));
        Interval B = mul_interval(new_interval(ca*param_down,ca*param_up),new_interval(cm*param_down,cm*param_up));
        Interval C = sub_interval(new_interval_v(u),new_interval(ch*param_down,ch*param_up));
        Interval D = mul_interval(B,C);
        Interval E = add_interval(A,D);
        rv = add_interval(disturbance_interval,E);
    }
    else if (dim ==3)
    {
        // theta' = v * (cos(beta)/(lf+lr)) * tan(delta)
        Interval mult = new_interval_v(1.0 /(lf +lr));
        Interval A = mul_interval(v,mult);
        Interval delt = new_interval_v(delta);
        Interval tan = div_interval(sin_interval(delt),cos_interval(delt));
        Interval tan2 = mul_interval(A,tan);
        rv = add_interval(disturbance_interval,tan2);

    }
    else 
    {
        printf("Error: Invalid Dimension");
        exit(0);
    }

    return isMin ? rv.min : rv.max;

}
#endif