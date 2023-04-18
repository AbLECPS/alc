#include <iostream>

extern "C"
{
     double getSimulatedSafeTime(double start[3],double heading,double speed);
}

int main(void)
{
    
    
    double startState[4] = {-1960.9, 42.01, 0.175};
    double control_input[2] = {11.21,0.335};

    double uh = control_input[0];
    double uv = control_input[1];

    getSimulatedSafeTime(startState,uh,uv);

    return 0;
}