#include <iostream>


const int max_hyper_rectangles = 2000;
 // num_rects
int rects = 0;
int rects2 = 0;
bool plot_all= true;

extern "C"
{
    #include "simulate_obstacle.h"
    HyperRectangle runReachability_obstacle_vis(REAL* start, REAL simTime, REAL wallTimeMs, REAL startMs,REAL v_x, REAL v_y, HyperRectangle VisStates[],int  *total_intermediate,int max_intermediate,bool plot);
    double getSimulatedSafeTime(double start[2],double v_x, double v_y);
    HyperRectangle hr_list[max_hyper_rectangles];
	HyperRectangle hr_list2[max_hyper_rectangles]; 
    void println(HyperRectangle* r);
}

int main(void)
{
    

    double startState[2] = {0.0, 0.0};
    double v_u[2] = {0.1,0.1};
    double v_x = v_u[0];
    double v_y = v_u[1];


    double simTime = 2.0; 
    int runtimeMs = 10; // in ms

    HyperRectangle reach_hull = runReachability_obstacle_vis(startState, simTime, runtimeMs, 0,v_x,v_y,hr_list,&rects,max_hyper_rectangles,plot_all);
	HyperRectangle reach_hull2 = runReachability_obstacle_vis(startState, simTime, runtimeMs, 0,v_x+0.1,v_y+0.1,hr_list2,&rects2,max_hyper_rectangles,plot_all);



    // // print the hull 
	std::cout << "rects count:" << rects <<std::endl;
    std::cout << "rects2 count:" << rects <<std::endl;

	for(int i= 0; i<10; i++)
    {
		println(&hr_list[i]);
    }

	std::cout<<"\n\n States 2 \n";

	for(int i= 0; i<10; i++)
    {
		println(&hr_list2[i]);
    }

	std::cout<< "\n\n Final States\n" << std::endl;
	println(&reach_hull);
	println(&reach_hull2);
    return 0;
}