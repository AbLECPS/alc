#include "dynamics_bicycle.h"
#include "dynamics_bicycle_uncertainty.h"
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

// parameter uncertainty here is a percentage so .1 corresponds to 10%.

// parameters that I tried
// double params[5] = {-0.0442, 12.8913, 0.8323,0.495,0.495};
// double params[5] = {-0.0119, 35.3912, 0.7818,0.495,0.495};
// double params[5] = {-0.0104, 15.3912, 0.7710,0.495,0.495};
// double params[5] = {0.0305, -8.0460, 0.7314,0.495,0.495};
// double params[5] = {-0.0301, -0.7323, -1.4485,0.495,0.495};
// double params[5] = {-0.0233, 0.0535, -3.8231,0.495,0.495};
// double params[5] = {-0.0229,  0.0511, -3.8231,0.495,0.495};
// double params[5] = {-0.0144, 1.0945, -1.0387,0.495,0.495};
// double params[5] = {-0.0160, 0.7924, -1.5859,0.495,0.495};
// double params[5] = {-0.0157, 0.7924, -1.5859,0.495,0.495};
// double params[5] = {-0.0229,  0.0511, -3.8231,0.495,0.495};
// double params[5] = {0.0000, 57.9958, -102.3728,0.495,0.495};
// double params[5] = {1e2 * 0.000000059825653, 1e2 *-0.579958155263372, 1e2 *-1.023728021236451,0.495,0.495};
// double params[5] = {1e2 * -0.000000056519793, 1e2 *-1.073515059199965, 1e2 *-1.361013342021826,0.495,0.495};
// double params[5] = {1e2 * 0.000000044578216, 1e2 * -0.6115101759186905, 1e2 * -1.222455828200963,0.495,0.495};
// double params[5] = {0.106420143929253, -2.755542141815733, 0.875436712647847,0.495,0.495};
// double params[5] = {0.001692552377488, -8.433264269018398, 0.000771663310643,0.495,0.495};
// double params[5] = {1e3 * 0.004008890653266, 1e3 *-8.424761078995289, 1e3 *0.000771914679619,0.495,0.495};
// double params[5] = {-0.0002, 795.0573, 0.7628,0.495,0.495};
// double params[5] = { -0.0013, 775.0573, 0.7627,0.495,0.495};

// original iver parameters
// double params[5] = {-0.0002, 780.0573, 0.7628,0.495,0.495};




// double params[5] = {0.000015467287802, -5.617669223427162, -4.850974295480865, 0.943756952661831, -0.181450649668541};
// double params[5] = {0.000016749649960,  -5.323728100228201,  -4.545338451400650, -0.180067739901494, 0.939757289761688};

// new experiments
// double params[5] = {-0.020669173511418,  -0.079504201628093,  -6.852899086492649, 0.489909817721430, 0.489140624999999}; not great
// double params[5] = {-0.000016749649960, -5.323728100228201, -4.545338451400650, -0.180067739901494, 0.939757289761688};  not great
// double params[5] = {0.235796018712068, -0.134468807978893, 6.329318823725194, 0.350492427349491, 0.350492456787161};

double params[5] = {0.0677,7.2439, 0.7941,0.2285,0.169};
// double params[5] = {0.012446703991232, -5.901821945662918, 0.909087562780645, 0.365847753521268, 0.365847755125654}; not bad 
// double params[5] = {0.012458478683986, -25.006973234143949, 0.805695287032192, 2.492270460305464, -1.783396831965558}; not bad
// double params[5] = {1e2*0.000056696166703, 1e2*-1.320737998722444, 1e2*0.007787132719015, 1e2*0.339466613789476, 1e2*-0.332099876591427}; not great
// double params[5] = {1e2*0.026835603833319, 1e2*-0.068420102448836, 1e2*11.928701397472215, 1e2*0.354395206257008, 1e2*0.354395937661522}; straight up bad but pretty interesting
// double params[5] = {1e2*0.001760192864847, 1e2*0.002563199618044, 1e2*52.015779886364328, 1e2*3.987212276688680,1e2*-1.769036749684242}; if you though the last one was mad, well this one is even wilder

double get_derivative_bounds_bicycle_uncertainty(HyperRectangle* rect, int faceIndex,REAL heading_input, REAL throttle, REAL parameter_uncertainty,REAL disturbance[][2])
{

    REAL u = throttle;
    REAL delta = heading_input;

    REAL ca = params[0];      //1.633;
    REAL cm = params[1];      //0.2;
    REAL ch = params[2];    //4; // These params come from sysid
    REAL lf = params[3];
    REAL lr = params[4];

    REAL param_up = (1.0+ parameter_uncertainty);
    REAL param_down = (1.0 - parameter_uncertainty);

    int dim = faceIndex / 2;
	bool isMin = (faceIndex % 2) == 0;

    Interval rv = new_interval_v(0);
    Interval v = rect->dims[2];
    Interval theta = rect->dims[3];

    // this interval will be used to add disturbances to each of the dimensions
    Interval disturbance_interval = new_interval(disturbance[dim][0],disturbance[dim][1]);

    if( dim == 0 ) 
    {
        // x' = v * cos(theta + beta)
        Interval A = mul_interval(v,cos_interval(theta));
        rv = add_interval(disturbance_interval,A);

    }
    else if(dim == 1)
    {
        // y' = v * sin(theta + beta)
        Interval A = mul_interval(v,sin_interval(theta));
        rv = add_interval(disturbance_interval,A);
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
        // printf("%f %f", disturbance_interval.min,disturbance_interval.max);
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
