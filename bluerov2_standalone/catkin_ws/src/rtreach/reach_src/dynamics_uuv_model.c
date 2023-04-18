#include "dynamics_uuv.h"
#include "dynamics_uuv_model.h"
#include "util.h"
#include "interval.h"

#include <stdlib.h>
#include <stdbool.h>
#include <stdio.h>
#include <math.h>

#ifdef DYNAMICS_UUV_MODEL

// Dynamical model for the UUV. The model was obtained using black box system identification techniques 
// With three states and two inputs. We neglect the depth of the uuv and assume it to be constant
// The form of the equations are

// X' = AX + BU

// X = [x;y;yaw] where x and y are the euclidean position, and yaw is the vehicle's orientation 
// U = [Heading, speed] where heading (degrees) and speed (m/s) are the set points for the  


// declare the matrices here to ease the replacement of new sys ID models

double A[3][3] = {{-0.000313916490540, 0.001061150070239, -3.348005730889027},
                {0.000039638097361, -0.000135859890048, -0.376038617601684},
                {0.000001532345552,  -0.000006491609986,  -0.538081047764798}};

double B[3][2] = {{0.050281804725666, 0.861318274378801},
               {0.017881793171043,  0.448247665566177},
               {0.009425493250813,   0.012325243536305}};

// implement the derivative using interval arithmetic
double get_derivative_bounds_uuv(HyperRectangle* rect, int faceIndex,REAL heading, REAL speed)
{
    int dim = faceIndex / 2;
	bool isMin = (faceIndex % 2) == 0;

    // Interval rv.min = rv.max =0
    Interval rv = new_interval_v(0);

    Interval x = rect->dims[0];
    Interval y = rect->dims[1];
    Interval yaw = rect->dims[2];
    Interval uh = new_interval_v(heading);
    Interval uv = new_interval_v(speed);

    
    
    Interval a1,a2,a3,b1,b2; // declare intervals for contants



    if( dim == 0 ) 
    {
        a1 = mul_interval(new_interval_v(A[0][0]),x);
        a2 = mul_interval(new_interval_v(A[0][1]),y);
        a3 = mul_interval(new_interval_v(A[0][2]),yaw);
        b1 = mul_interval(new_interval_v(B[0][0]),uh);
        b2 = mul_interval(new_interval_v(B[0][1]),uv);
        rv = add_interval(add_interval(add_interval(add_interval(a1,a2),a3),b1),b2);
    }
    else if(dim == 1)
    {
        a1 = mul_interval(new_interval_v(A[1][0]),x);
        a2 = mul_interval(new_interval_v(A[1][1]),y);
        a3 = mul_interval(new_interval_v(A[1][2]),yaw);
        b1 = mul_interval(new_interval_v(B[1][0]),uh);
        b2 = mul_interval(new_interval_v(B[1][1]),uv);
        rv = add_interval(add_interval(add_interval(add_interval(a1,a2),a3),b1),b2);
    }
    else if(dim ==2)
    {
        a1 = mul_interval(new_interval_v(A[2][0]),x);
        a2 = mul_interval(new_interval_v(A[2][1]),y);
        a3 = mul_interval(new_interval_v(A[2][2]),yaw);
        b1 = mul_interval(new_interval_v(B[2][0]),uh);
        b2 = mul_interval(new_interval_v(B[2][1]),uv);
        rv = add_interval(add_interval(add_interval(add_interval(a1,a2),a3),b1),b2);
    }
    else 
    {
        printf("Error: Invalid Dimension");
        exit(0);
    }

    return isMin ? rv.min : rv.max;

}
#endif